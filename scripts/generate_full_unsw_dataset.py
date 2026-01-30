
import csv
import pickle
import numpy as np
import sys
import os
from collections import defaultdict
import math

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RAW_CSV = "data/master_dataset.csv"       # The 15GB Raw File
GT_CSV = "UNSW-NB15_GT.csv"               # The 83MB Label File
OUT_DIR = "data/unswnb15_full"            # Output folder
OUT_FILE = f"{OUT_DIR}/flows_all.pkl"     # Final Output
MAX_PACKETS = 32                          # Mamba Input Size
TIMEOUT = 60.0                            # Usage: Split flows if idle > 60s (Standard)

# ==============================================================================
# LOGIC
# ==============================================================================

def load_ground_truth(gt_path):
    print(f"Loading Ground Truth from {gt_path}...")
    gt_lookup = defaultdict(list)
    count = 0
    
    with open(gt_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        try: header = next(reader)
        except: return gt_lookup
        
        # GT Col Indices (Based on inspection)
        # Start time, Last time, Attack category, Attack subcategory, Protocol, Source IP, Source Port, Destination IP, Destination Port...
        # 0: Start, 1: End, 2: Cat, 4: Proto, 5: SrcIP, 6: SrcPort, 7: DstIP, 8: DstPort
        
        for row in reader:
            try:
                st = float(row[0])
                et = float(row[1])
                label = row[2].strip()
                proto = row[4].lower().strip()
                src_ip = row[5].strip()
                src_port = row[6].strip()
                dst_ip = row[7].strip()
                dst_port = row[8].strip()
                
                # key = 5-tuple text
                key = (src_ip, dst_ip, src_port, dst_port, proto)
                
                # Store interval matches
                gt_lookup[key].append({
                    'start': st,
                    'end': et,
                    'label': label
                })
                count += 1
            except: continue
            
    print(f"Loaded {count} Attack Events. Lookup Table Ready.")
    return gt_lookup

def get_label_for_flow(flow_key, flow_start_time, flow_end_time, gt_lookup):
    """
    Check if flow overlaps with any GT attack interval.
    If matching Tuples found, check Time.
    """
    if flow_key in gt_lookup:
        events = gt_lookup[flow_key]
        for e in events:
            # Overlap check: (GtStart <= FlowEnd) and (GTEnd >= FlowStart)
            # Typically attacks are subsets of flows or match exactly.
            # Loose check: If flow started during attack window?
            # Strict check: Overlap.
            if (e['start'] <= flow_end_time + 1.0) and (e['end'] >= flow_start_time - 1.0):
                return e['label']
    return "Benign" 

def process_raw(raw_path, gt_lookup):
    print(f"Processing Raw Packets from {raw_path}...")
    
    proto_map = {'icmp': 1, 'tcp': 6, 'udp': 17, 'other': 0}
    
    # Flow Cache: Key -> List of Packets
    active_flows = defaultdict(list)
    finished_flows = []
    
    packet_count = 0
    
    # Output Buffer
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    with open(raw_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Indices from master_dataset.csv head:
        # frame.time_epoch, ip.src, ip.dst, tcp.srcport, udp.srcport, tcp.dstport, udp.dstport, ip.proto, frame.len, tcp.flags
        # 0: time
        # 1: src
        # 2: dst
        # 3: tcp_src
        # 4: udp_src
        # 5: tcp_dst
        # 6: udp_dst
        # 7: proto (int) - Wait, in master_dataset it was '6' (string int)
        # 8: len
        # 9: flags
        
        for row in reader:
            packet_count += 1
            if packet_count % 1_000_000 == 0:
                print(f"Processed {packet_count/1e6:.1f}M packets... (Collected flows: {len(finished_flows)})")
            
            try:
                ts = float(row[0])
                src = row[1]
                dst = row[2]
                
                # Ports handle
                sport = row[3] if row[3] else row[4]
                dport = row[5] if row[5] else row[6]
                if not sport: sport = "0"
                if not dport: dport = "0"
                
                proto_raw = row[7] # likely '6', '17'
                try: p_idx = int(proto_raw)
                except: p_idx = 0
                
                # Normalize proto string for Key match (GT uses 'tcp', 'udp')
                if p_idx == 6: p_str = 'tcp'
                elif p_idx == 17: p_str = 'udp'
                elif p_idx == 1: p_str = 'icmp'
                else: p_str = 'other'
                
                pkt_len = float(row[8])
                flags = row[9]
                try: flag_int = int(flags, 16) if 'x' in flags else int(flags)
                except: flag_int = 0
                
                # Key for Logic
                flow_key = (src, dst, sport, dport, p_str)
                
                # Add to flow
                # Packet: [proto_int, len, flags, ts, dir(0)]
                pkt = [p_idx, pkt_len, flag_int, ts, 0]
                
                active_flows[flow_key].append(pkt)
                
                # Splitting Logic (OPTIONAL): 
                # If doing simple full capture, just keep appending? 
                # Real systems split on timeout. For simplicity/memory, 
                # if > MAX_PACKETS, we could assume flow is "done" for *this model's view*
                # but we need correct Label logic which depends on Time.
                # Let's keep first 32 per flow, ignore rest, but keep tracking time for labeling?
                # Optimization: Only store first 32. Update 'last_seen' timestamp separately.
                
            except ValueError: continue
            
    print("Packet Processing Complete. Generating Features...")
    
    # Convert Active Flows to Final Features
    final_output = []
    
    for key, pkts in active_flows.items():
        if not pkts: continue
        
        # Time Window
        start_ts = pkts[0][3]
        end_ts = pkts[-1][3]
        
        # match label
        label_str = get_label_for_flow(key, start_ts, end_ts, gt_lookup)
        label_int = 1 if label_str != "Benign" else 0
        
        # Extract Features (First 32)
        pkts_slice = pkts[:MAX_PACKETS]
        prev_time = start_ts
        
        vec_proto = []
        vec_len = []
        vec_flags = []
        vec_iat = []
        vec_dir = []
        
        for p in pkts_slice:
            cur_time = p[3]
            iat = max(0, cur_time - prev_time)
            
            # Log Scales
            log_len = math.log1p(p[1])
            log_iat = math.log1p(iat + 1e-6)
            
            vec_proto.append(p[0])
            vec_len.append(log_len)
            vec_flags.append(p[2])
            vec_iat.append(log_iat)
            vec_dir.append(p[4])
            
            prev_time = cur_time
            
        # Pad
        L = len(vec_proto)
        if L < MAX_PACKETS:
            pad = MAX_PACKETS - L
            vec_proto += [0]*pad
            vec_len += [0]*pad
            vec_flags += [0]*pad
            vec_iat += [0]*pad
            vec_dir += [0]*pad
            
        # Feature Matrix
        features = np.stack([vec_proto, vec_len, vec_flags, vec_iat, vec_dir], axis=1) # (32, 5)
        
        flow_record = {
            'features': features.astype(np.float32),
            'label': label_int,
            'label_str': label_str,
            'key': key
        }
        final_output.append(flow_record)
        
    print(f"Generated {len(final_output)} Flows.")
    print(f"Saving to {OUT_FILE}...")
    with open(OUT_FILE, 'wb') as f:
        pickle.dump(final_output, f)
    print("Done.")

if __name__ == "__main__":
    gt = load_ground_truth(GT_CSV)
    process_raw(RAW_CSV, gt)

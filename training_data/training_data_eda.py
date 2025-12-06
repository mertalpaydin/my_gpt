import os
from pathlib import Path
import pyarrow.parquet as pq
import tiktoken
import numpy as np

def main():
    script_dir = Path(__file__).parent
    parquet_path = script_dir / "fineweb_edu_sample" / "sample" / "10BT" / "000_00000.parquet"
    
    print("=" * 70)
    print("FINEWEB-EDU TRAINING DATA ANALYSIS")
    print("=" * 70)
    
    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        print("Run download_training_data.py first.")
        return
    
    print(f"\nLoading: {parquet_path}")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    print("\n" + "=" * 70)
    print("BASIC STATS")
    print("=" * 70)
    
    file_size_bytes = parquet_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Parquet file size:    {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"Number of documents:  {len(df):,}")
    print(f"Columns:              {list(df.columns)}")
    
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    print("\n" + "=" * 70)
    print("TEXT LENGTH STATISTICS (characters)")
    print("=" * 70)
    
    df['text_length'] = df['text'].str.len()
    
    print(f"Min length:           {df['text_length'].min():,}")
    print(f"Max length:           {df['text_length'].max():,}")
    print(f"Mean length:          {df['text_length'].mean():,.2f}")
    print(f"Median length:        {df['text_length'].median():,.2f}")
    print(f"Std deviation:        {df['text_length'].std():,.2f}")
    print(f"Total characters:     {df['text_length'].sum():,}")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(df['text_length'], p)
        print(f"  {p}th percentile:    {val:,.0f} chars")
    
    print("\n" + "=" * 70)
    print("TOKENIZATION STATS (GPT-2 BPE)")
    print("=" * 70)
    
    enc = tiktoken.get_encoding("gpt2")
    
    sample_size = min(1000, len(df))
    print(f"Sampling {sample_size} documents for token analysis...")
    
    sample_df = df.sample(n=sample_size, random_state=42)
    token_counts = []
    
    for text in sample_df['text']:
        tokens = enc.encode(text)
        token_counts.append(len(tokens))
    
    token_counts = np.array(token_counts)
    
    print(f"\nToken counts (sampled from {sample_size} docs):")
    print(f"Min tokens:           {token_counts.min():,}")
    print(f"Max tokens:           {token_counts.max():,}")
    print(f"Mean tokens:          {token_counts.mean():,.2f}")
    print(f"Median tokens:        {np.median(token_counts):,.2f}")
    
    estimated_total_tokens = (token_counts.mean() * len(df))
    print(f"\nEstimated total tokens in dataset: {estimated_total_tokens:,.0f} (~{estimated_total_tokens/1e6:.1f}M)")
    
    print(f"\nToken percentiles:")
    for p in percentiles:
        val = np.percentile(token_counts, p)
        print(f"  {p}th percentile:    {val:,.0f} tokens")
    
    docs_under_1024 = sum(1 for tc in token_counts if tc < 1024) / len(token_counts) * 100
    print(f"\nDocs under 1024 tokens: {docs_under_1024:.1f}%")
    
    print("\n" + "=" * 70)
    print("SAMPLE DOCUMENTS")
    print("=" * 70)
    
    for i, idx in enumerate(df.sample(3, random_state=123).index):
        row = df.loc[idx]
        text = row['text']
        preview = text[:500] + "..." if len(text) > 500 else text
        tokens = enc.encode(text)
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Index: {idx}")
        print(f"Length: {len(text):,} chars | {len(tokens):,} tokens")
        
        for col in df.columns:
            if col not in ['text', 'text_length']:
                print(f"{col}: {row[col]}")
        
        print(f"\nText preview:\n{preview}")
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("MEMORY USAGE")
    print("=" * 70)
    
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"DataFrame in memory:  {memory_mb:.2f} MB")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

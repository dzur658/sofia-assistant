import cudf
import time

def clean_opus_gpu(input_file, output_file):
    print(f"Loading {input_file} directly into VRAM...")
    start_time = time.time()
    
    # 1. Load the raw text file as a single-column GPU DataFrame
    df = cudf.read_csv(input_file, header=None, names=['text'], sep='\t', quoting=3)
    
    print(f"Loaded {len(df)} rows. Starting GPU regex pipeline...")

    # 2. Drop watermarks completely
    # We use a boolean mask to filter out rows containing these strings
    watermarks = r'opensubtitles|subdivx|argenteam|sincronización por|traducido por|subtítulos descargados'
    
    # Evaluate against a lowercased temporary series, 
    # but apply the mask to preserve the original casing in df['text']
    mask = ~df['text'].str.lower().str.contains(watermarks, regex=True)
    df = df[mask]

    # 3. Strip HTML tags (<i>, <b>, <font...>)
    df['text'] = df['text'].str.replace(r'<[^>]+>', '', regex=True)
    
    # 4. Strip Closed Captioning [Risas], (Música)
    df['text'] = df['text'].str.replace(r'\[.*?\]|\(.*?\)', '', regex=True)
    
    # 5. Remove ALL-CAPS speaker labels (e.g., "JUAN: Hola")
    df['text'] = df['text'].str.replace(r'^[A-ZÁÉÍÓÚÑ\s0-9]+:\s*', '', regex=True)
    
    # 6. Handle the Ellipsis Disease (Frame splits)
    # cuDF's regex engine (RE2) supports backreferences (\1)
    df['text'] = df['text'].str.replace(r'\.\.\.\s*([a-záéíóúñ])', r' \1', regex=True)
    df['text'] = df['text'].str.replace(r'\s*\.\.\.\s*', ' ', regex=True)
    
    # 7. Normalize leading hyphens to standard Spanish dialogue dashes
    df['text'] = df['text'].str.replace(r'^-\s*', '— ', regex=True)
    
    # 8. Cleanup extra spaces and strip
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Drop any rows that became empty after cleaning
    df = df[df['text'] != ""]

    print("Regex complete. Formatting chunks...")

    # Add after all cleaning
    print("Deduplicating entries...")

    initial_count = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f"Removed {initial_count - len(df)} duplicate rows")

    # NOTE ON CHUNKING: 
    # Because you are moving to a DataFrame format, chunking logic is slightly 
    # different than standard string accumulation. If you need strict 2048-token 
    # blocks, you would pull the cleaned column back to the CPU (`df.to_pandas()`) 
    # here to run the fast token-counting aggregator, as string concatenation 
    # across dynamic row boundaries is less efficient on a strict GPU grid.
    
    # For now, saving the pristine cleaned lines:
    df.to_parquet(output_file, compression='snappy',index=False)
    
    print(f"Finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    clean_opus_gpu("es.txt", "opus_es_cleaned_gpu.parquet")
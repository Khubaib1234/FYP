from sentence_transformers import SentenceTransformer

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Downloading/loading model: {model_name} ...")
    
    model = SentenceTransformer(model_name)
    
    print("Model loaded successfully!")
    print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

if __name__ == "__main__":
    main()

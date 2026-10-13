#!/usr/bin/env python3
"""
Test script to verify metadata is properly preserved in ChromaDB ingestion and retrieval.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_chunks"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def test_metadata_preservation():
    """Test that metadata is properly preserved during ingestion and retrieval."""
    print("Testing metadata preservation in ChromaDB...")
    
    try:
        # Connect to ChromaDB
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        col = client.get_collection(COLLECTION_NAME)
        
        # Test query
        test_query = "research methodology"
        model = SentenceTransformer(EMBED_MODEL)
        q_emb = model.encode(test_query, convert_to_numpy=True).tolist()
        
        # Retrieve with metadata
        res = col.query(
            query_embeddings=[q_emb],
            n_results=3,
            include=["documents", "metadatas", "distances"],
            where={"source": {"$eq": "thesis_text_chunks_v1"}}
        )
        
        print(f"\nQuery: '{test_query}'")
        print(f"Raw ChromaDB response structure:")
        print(f"- documents: {type(res.get('documents'))}, length: {len(res.get('documents', []))}")
        print(f"- metadatas: {type(res.get('metadatas'))}, length: {len(res.get('metadatas', []))}")
        print(f"- distances: {type(res.get('distances'))}, length: {len(res.get('distances', []))}")
        
        # Extract results (ChromaDB returns nested lists)
        documents = res.get("documents", [[]])[0] if res.get("documents") else []
        metadatas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        distances = res.get("distances", [[]])[0] if res.get("distances") else []
        
        print(f"\nExtracted results:")
        print(f"- {len(documents)} documents")
        print(f"- {len(metadatas)} metadata objects")
        print(f"- {len(distances)} distances")
        
        if not documents:
            print("❌ No documents found! Check if data is ingested properly.")
            return False
            
        if len(documents) != len(metadatas):
            print(f"❌ Mismatch: {len(documents)} documents vs {len(metadatas)} metadata objects")
            return False
            
        print("\n" + "="*80)
        print("DETAILED RESULTS:")
        print("="*80)
        
        all_metadata_valid = True
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
            print(f"\nResult #{i}:")
            print(f"Distance: {dist:.4f}")
            print(f"Document preview: {doc[:200]}...")
            print(f"Metadata type: {type(meta)}")
            print(f"Metadata content: {meta}")
            
            # Check if metadata has expected fields
            if not isinstance(meta, dict):
                print(f"❌ Metadata is not a dict: {type(meta)}")
                all_metadata_valid = False
                continue
                
            expected_fields = ["source", "doc_id", "chunk_index", "length"]
            missing_fields = [field for field in expected_fields if field not in meta]
            
            if missing_fields:
                print(f"❌ Missing metadata fields: {missing_fields}")
                all_metadata_valid = False
            else:
                print(f"✅ All expected metadata fields present")
                
            # Generate citation
            src = meta.get("source", "unknown")
            doc_id = meta.get("doc_id", "")
            chunk_idx = meta.get("chunk_index", "n/a")
            citation = f"[{src}; {doc_id}; chunk_index={chunk_idx}]"
            print(f"Citation: {citation}")
            
        print("\n" + "="*80)
        
        if all_metadata_valid:
            print("✅ SUCCESS: All metadata is properly preserved!")
            return True
        else:
            print("❌ FAILURE: Some metadata issues detected.")
            return False
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def inspect_collection_sample():
    """Inspect a few random documents from the collection to see their metadata."""
    print("\nInspecting collection sample...")
    
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        col = client.get_collection(COLLECTION_NAME)
        
        # Get a few random documents
        result = col.get(
            limit=5,
            include=["documents", "metadatas"]
        )
        
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        
        print(f"Sample of {len(documents)} documents from collection:")
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            print(f"\nSample #{i}:")
            print(f"Document: {doc[:100]}...")
            print(f"Metadata: {meta}")
            
    except Exception as e:
        print(f"Error inspecting collection: {e}")

if __name__ == "__main__":
    print("ChromaDB Metadata Test")
    print("=" * 50)
    
    # Test metadata preservation
    success = test_metadata_preservation()
    
    # Inspect collection sample
    inspect_collection_sample()
    
    print("\n" + "="*50)
    if success:
        print("✅ OVERALL: Metadata handling is working correctly!")
    else:
        print("❌ OVERALL: Issues detected with metadata handling.")
        print("\nTroubleshooting tips:")
        print("1. Make sure ChromaDB server is running")
        print("2. Verify data was ingested with ingest_chroma.py")
        print("3. Check that the collection name matches")
        print("4. Ensure the 'source' filter value is correct")

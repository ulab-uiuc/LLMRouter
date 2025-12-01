"""
Index documentation files for LLM search (ChromaDB 1.3.0 compatible)
"""
import os
import re
from pathlib import Path
import chromadb
import markdown
from bs4 import BeautifulSoup

class DocumentationIndexer:
    def __init__(self, docs_dir: str = "../docs", db_path: str = "./chroma_db"):
        self.docs_dir = Path(docs_dir)
        self.db_path = db_path
        
        # Initialize ChromaDB client for version 1.3.0
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection("documentation")
            print("Deleted existing collection")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name="documentation",
            metadata={"hnsw:space": "cosine"}
        )
    
    def extract_text_from_markdown(self, md_content: str) -> str:
        """Convert markdown to plain text"""
        html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def extract_metadata(self, filepath: Path, content: str) -> dict:
        """Extract metadata from markdown file"""
        metadata = {
            'source': str(filepath.relative_to(self.docs_dir)),
            'title': filepath.stem.replace('-', ' ').title()
        }
        
        # Try to extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1)
        
        return metadata
    
    def index_documentation(self):
        """Index all markdown files in docs directory"""
        print(f"Indexing documentation from {self.docs_dir}...")
        
        documents = []
        metadatas = []
        ids = []
        
        doc_count = 0
        
        # Walk through all markdown files
        for md_file in self.docs_dir.rglob("*.md"):
            if md_file.name.startswith('.'):
                continue
            
            print(f"Processing: {md_file.relative_to(self.docs_dir)}")
            
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract metadata
                metadata = self.extract_metadata(md_file, content)
                
                # Convert to plain text
                text = self.extract_text_from_markdown(content)
                
                # Skip if text is too short
                if len(text.strip()) < 50:
                    continue
                
                # Chunk the text
                chunks = self.chunk_text(text)
                
                # Add each chunk to the collection
                for i, chunk in enumerate(chunks):
                    doc_id = f"{metadata['source']}_{i}"
                    documents.append(chunk)
                    metadatas.append({
                        **metadata,
                        'chunk_id': str(i),
                        'total_chunks': str(len(chunks))
                    })
                    ids.append(doc_id)
                    doc_count += 1
            
            except Exception as e:
                print(f"Error processing {md_file}: {e}")
        
        # Add all documents to ChromaDB in batches
        if documents:
            print(f"\nAdding {len(documents)} chunks to database...")
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                print(f"Adding batch {i//batch_size + 1} ({i} to {end_idx})...")
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
        
        print(f"\n✓ Indexed {doc_count} document chunks from {len(list(self.docs_dir.rglob('*.md')))} files")
        return doc_count

if __name__ == "__main__":
    indexer = DocumentationIndexer()
    indexer.index_documentation()
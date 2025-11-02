"""
RAG (Retrieval Augmented Generation) implementation for restaurant knowledge base.

This module provides semantic search over restaurant documents using ChromaDB.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from Data import helpers as data_helpers


class RestaurantRAG:
    """RAG system for restaurant knowledge base."""

    def __init__(self, persist_directory: str = "Data/chroma_db"):
        """
        Initialize RAG system with ChromaDB.

        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self._initialized:
            return

        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="restaurants",
            metadata={"description": "Restaurant information and documents"}
        )

        self._initialized = True

    def index_restaurants(self, restaurants_path: str = "Data/restaurants.json") -> int:
        """
        Index restaurant data into the vector database.

        Args:
            restaurants_path: Path to restaurants JSON file

        Returns:
            Number of documents indexed
        """
        self.initialize()

        # Load restaurant data
        restaurants = data_helpers.load_restaurants(restaurants_path)

        if not restaurants:
            return 0

        # Prepare documents for indexing
        documents = []
        metadatas = []
        ids = []

        for restaurant in restaurants:
            # Create rich text representation for embedding
            doc_text = self._create_document_text(restaurant)
            documents.append(doc_text)

            # Store metadata
            metadatas.append({
                "id": restaurant.get("id", ""),
                "name": restaurant.get("name", ""),
                "cuisine": restaurant.get("cuisine", ""),
                "location": restaurant.get("location", ""),
                "price_range": restaurant.get("price_range", ""),
                "rating": float(restaurant.get("rating", 0.0))
            })

            ids.append(restaurant.get("id", f"restaurant_{len(ids)}"))

        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return len(documents)

    def _create_document_text(self, restaurant: Dict[str, Any]) -> str:
        """
        Create searchable text document from restaurant data.

        Args:
            restaurant: Restaurant data dictionary

        Returns:
            Formatted text for embedding
        """
        parts = [
            f"Restaurant: {restaurant.get('name', 'Unknown')}",
            f"Cuisine: {restaurant.get('cuisine', 'Various')}",
            f"Location: {restaurant.get('location', 'Unknown')} - {restaurant.get('address', '')}",
            f"Price Range: {restaurant.get('price_range', '')} ({restaurant.get('average_price_per_person', '')})",
            f"Rating: {restaurant.get('rating', 0)}/5.0"
        ]

        # Add features
        features = restaurant.get('features', [])
        if features:
            parts.append(f"Features: {', '.join(features)}")

        # Add popular dishes
        dishes = restaurant.get('popular_dishes', [])
        if dishes:
            parts.append(f"Popular dishes: {', '.join(dishes)}")

        # Add dietary options
        dietary = restaurant.get('dietary_options', [])
        if dietary:
            parts.append(f"Dietary options: {', '.join(dietary)}")

        # Add boolean features
        if restaurant.get('outdoor_seating'):
            parts.append("Outdoor seating available")
        if restaurant.get('private_dining'):
            parts.append("Private dining available")
        if restaurant.get('reservations_required'):
            parts.append("Reservations required")
        if restaurant.get('takeout_available'):
            parts.append("Takeout available")
        if restaurant.get('delivery_available'):
            parts.append("Delivery available")

        return "\n".join(parts)

    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base using semantic search.

        Args:
            query_text: Natural language query
            n_results: Number of results to return

        Returns:
            List of matching restaurants with relevance scores
        """
        self.initialize()

        # Check if collection is empty
        if self.collection.count() == 0:
            # Auto-index if empty
            indexed_count = self.index_restaurants()
            if indexed_count == 0:
                return []

        # Perform semantic search
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count())
        )

        # Format results
        formatted_results = []
        for i, doc_id in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else None

            formatted_results.append({
                'id': metadata.get('id'),
                'name': metadata.get('name'),
                'cuisine': metadata.get('cuisine'),
                'location': metadata.get('location'),
                'price_range': metadata.get('price_range'),
                'rating': metadata.get('rating'),
                'relevance_score': 1.0 - distance if distance is not None else None,
                'document': results['documents'][0][i]
            })

        return formatted_results

    def reset_index(self) -> None:
        """Reset the vector database (delete and recreate)."""
        if self.client and self.collection:
            self.client.delete_collection("restaurants")
            self.collection = self.client.create_collection(
                name="restaurants",
                metadata={"description": "Restaurant information and documents"}
            )


# Global RAG instance
_rag_instance: Optional[RestaurantRAG] = None


def get_rag_instance() -> RestaurantRAG:
    """
    Get or create the global RAG instance.

    Returns:
        RestaurantRAG instance
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RestaurantRAG()
    return _rag_instance


def search_knowledge_base(
    query: str,
    limit: int = 5,
    use_aws: bool = True,
    aws_weight: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Search the restaurant knowledge base using hybrid RAG (local + AWS).

    This function combines results from:
    1. Local ChromaDB (structured restaurant data)
    2. AWS Bedrock Knowledge Base (semantic documents from S3)

    Args:
        query: Natural language search query
        limit: Maximum number of results
        use_aws: Whether to include AWS Bedrock KB results
        aws_weight: Weight for AWS results in hybrid scoring (0-1)

    Returns:
        List of matching restaurants with combined relevance scores
    """
    results = []

    # Get local ChromaDB results
    rag = get_rag_instance()
    local_results = rag.query(query, n_results=limit)

    # Add source tag
    for result in local_results:
        result['source'] = 'local'
        results.append(result)

    # Get AWS Bedrock KB results if enabled
    if use_aws:
        try:
            from aws_kb import search_aws_knowledge_base

            aws_results = search_aws_knowledge_base(query, max_results=limit)

            # Process AWS results and merge with local
            for aws_result in aws_results:
                # Try to match AWS result to a restaurant in local data
                content = aws_result.get('content', '')
                metadata = aws_result.get('metadata', {})

                # Extract restaurant info from AWS content/metadata
                restaurant_info = {
                    'id': metadata.get('restaurant_id', 'aws_unknown'),
                    'name': metadata.get('restaurant_name', 'From AWS KB'),
                    'cuisine': metadata.get('cuisine', ''),
                    'location': metadata.get('location', ''),
                    'price_range': metadata.get('price_range', ''),
                    'rating': float(metadata.get('rating', 0.0)),
                    'relevance_score': aws_result.get('score', 0.0),
                    'document': content,
                    'source': 'aws_bedrock'
                }

                results.append(restaurant_info)

        except Exception as e:
            logger.warning(f"AWS KB search failed, using local only: {e}")

    # Deduplicate and combine scores for restaurants found in both sources
    combined = {}
    for result in results:
        restaurant_id = result['id']

        if restaurant_id in combined:
            # Combine relevance scores using weighted average
            existing = combined[restaurant_id]
            existing_score = existing.get('relevance_score', 0.0) or 0.0
            new_score = result.get('relevance_score', 0.0) or 0.0

            if result['source'] == 'aws_bedrock':
                # AWS result gets its weight
                combined_score = (existing_score * (1 - aws_weight)) + (new_score * aws_weight)
            else:
                # Local result
                combined_score = (existing_score * aws_weight) + (new_score * (1 - aws_weight))

            combined[restaurant_id]['relevance_score'] = combined_score
            combined[restaurant_id]['source'] = 'hybrid'

            # Enrich with AWS content if available
            if result['source'] == 'aws_bedrock' and result.get('document'):
                combined[restaurant_id]['aws_content'] = result['document']
        else:
            combined[restaurant_id] = result

    # Convert back to list and sort by relevance
    final_results = list(combined.values())
    final_results.sort(key=lambda x: x.get('relevance_score', 0.0) or 0.0, reverse=True)

    return final_results[:limit]


# CLI for testing
if __name__ == "__main__":
    import sys

    rag = RestaurantRAG()

    if len(sys.argv) > 1:
        if sys.argv[1] == "index":
            print("Indexing restaurants...")
            count = rag.index_restaurants()
            print(f"Indexed {count} restaurants")

        elif sys.argv[1] == "reset":
            print("Resetting index...")
            rag.reset_index()
            print("Index reset complete")

        elif sys.argv[1] == "query":
            if len(sys.argv) < 3:
                print("Usage: python rag.py query <query_text>")
                sys.exit(1)

            query_text = " ".join(sys.argv[2:])
            print(f"Searching for: {query_text}\n")

            results = rag.query(query_text)
            if not results:
                print("No results found")
            else:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['name']} ({result['cuisine']})")
                    print(f"   Location: {result['location']}")
                    print(f"   Rating: {result['rating']}/5.0")
                    if result['relevance_score']:
                        print(f"   Relevance: {result['relevance_score']:.2%}")
                    print()
    else:
        print("Usage:")
        print("  python rag.py index          - Index restaurant data")
        print("  python rag.py reset          - Reset the index")
        print("  python rag.py query <text>   - Search the knowledge base")

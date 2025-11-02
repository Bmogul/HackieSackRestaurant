"""
AWS Bedrock Knowledge Base Integration.

This module provides access to restaurant semantic data stored in AWS S3
and indexed via Amazon Bedrock Knowledge Base.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logging.warning("boto3 not available. AWS features will be disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSKnowledgeBase:
    """
    AWS Bedrock Knowledge Base client for restaurant semantic data.

    This class interfaces with:
    - Amazon S3 (for document storage)
    - Amazon Bedrock Knowledge Bases (for semantic retrieval)
    """

    def __init__(
        self,
        knowledge_base_id: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "restaurant-docs/",
        region_name: str = "us-east-1"
    ):
        """
        Initialize AWS Knowledge Base client.

        Args:
            knowledge_base_id: Bedrock Knowledge Base ID
            s3_bucket: S3 bucket name containing restaurant documents
            s3_prefix: S3 key prefix for restaurant documents
            region_name: AWS region
        """
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 is required for AWS features. Install with: pip install boto3")

        # Load from environment variables if not provided
        self.knowledge_base_id = knowledge_base_id or os.getenv('AWS_KNOWLEDGE_BASE_ID')
        self.s3_bucket = s3_bucket or os.getenv('AWS_S3_BUCKET')
        self.s3_prefix = s3_prefix
        self.region_name = region_name

        # Initialize AWS clients
        try:
            self.bedrock_agent_runtime = boto3.client(
                'bedrock-agent-runtime',
                region_name=self.region_name
            )
            self.s3_client = boto3.client(
                's3',
                region_name=self.region_name
            )
            logger.info(f"AWS clients initialized for region: {self.region_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise

        self._verify_configuration()

    def _verify_configuration(self) -> None:
        """Verify AWS configuration."""
        if not self.knowledge_base_id:
            logger.warning("AWS_KNOWLEDGE_BASE_ID not set. Knowledge base queries will fail.")

        if not self.s3_bucket:
            logger.warning("AWS_S3_BUCKET not set. S3 document retrieval will fail.")
        else:
            logger.info(f"Using S3 bucket: {self.s3_bucket}/{self.s3_prefix}")

    def query_knowledge_base(
        self,
        query: str,
        max_results: int = 5,
        relevance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Query the AWS Bedrock Knowledge Base for semantic retrieval.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            relevance_threshold: Minimum relevance score (0-1)

        Returns:
            List of retrieved documents with metadata and scores
        """
        if not self.knowledge_base_id:
            logger.error("Knowledge Base ID not configured")
            return []

        try:
            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={
                    'text': query
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )

            # Parse and filter results
            results = []
            for item in response.get('retrievalResults', []):
                score = item.get('score', 0.0)

                # Filter by relevance threshold
                if score < relevance_threshold:
                    continue

                result = {
                    'content': item.get('content', {}).get('text', ''),
                    'score': score,
                    'location': item.get('location', {}),
                    'metadata': item.get('metadata', {})
                }

                results.append(result)

            logger.info(f"Retrieved {len(results)} results from AWS KB for query: '{query}'")
            return results

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS Bedrock error ({error_code}): {error_message}")
            return []
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []

    def retrieve_and_generate(
        self,
        query: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query knowledge base and generate response using Bedrock LLM.

        Args:
            query: Natural language query
            model_id: Bedrock model ID for generation
            max_results: Number of documents to retrieve

        Returns:
            Dictionary with generated response and source citations
        """
        if not self.knowledge_base_id:
            logger.error("Knowledge Base ID not configured")
            return {"response": "", "citations": []}

        try:
            response = self.bedrock_agent_runtime.retrieve_and_generate(
                input={
                    'text': query
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self.knowledge_base_id,
                        'modelArn': f'arn:aws:bedrock:{self.region_name}::foundation-model/{model_id}',
                        'retrievalConfiguration': {
                            'vectorSearchConfiguration': {
                                'numberOfResults': max_results
                            }
                        }
                    }
                }
            )

            output = response.get('output', {}).get('text', '')
            citations = response.get('citations', [])

            logger.info(f"Generated response with {len(citations)} citations")

            return {
                'response': output,
                'citations': citations,
                'session_id': response.get('sessionId')
            }

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"AWS Bedrock error ({error_code}): {error_message}")
            return {"response": f"Error: {error_message}", "citations": []}
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {e}")
            return {"response": f"Error: {str(e)}", "citations": []}

    def list_s3_documents(self) -> List[str]:
        """
        List all restaurant documents in S3 bucket.

        Returns:
            List of S3 object keys
        """
        if not self.s3_bucket:
            logger.error("S3 bucket not configured")
            return []

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=self.s3_prefix
            )

            documents = []
            for obj in response.get('Contents', []):
                key = obj.get('Key')
                if key and not key.endswith('/'):
                    documents.append(key)

            logger.info(f"Found {len(documents)} documents in S3")
            return documents

        except ClientError as e:
            logger.error(f"Error listing S3 documents: {e}")
            return []

    def download_s3_document(self, key: str) -> Optional[str]:
        """
        Download and read a document from S3.

        Args:
            key: S3 object key

        Returns:
            Document content as string, or None if error
        """
        if not self.s3_bucket:
            logger.error("S3 bucket not configured")
            return None

        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=key
            )

            content = response['Body'].read().decode('utf-8')
            logger.info(f"Downloaded document: {key} ({len(content)} bytes)")
            return content

        except ClientError as e:
            logger.error(f"Error downloading S3 document {key}: {e}")
            return None

    def upload_document(self, content: str, key: str) -> bool:
        """
        Upload a document to S3.

        Args:
            content: Document content
            key: S3 object key (will be prefixed with s3_prefix)

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_bucket:
            logger.error("S3 bucket not configured")
            return False

        full_key = f"{self.s3_prefix}{key}"

        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=full_key,
                Body=content.encode('utf-8')
            )
            logger.info(f"Uploaded document: {full_key}")
            return True

        except ClientError as e:
            logger.error(f"Error uploading document to S3: {e}")
            return False


# Global AWS KB instance
_aws_kb_instance: Optional[AWSKnowledgeBase] = None


def get_aws_kb_instance() -> Optional[AWSKnowledgeBase]:
    """
    Get or create the global AWS Knowledge Base instance.

    Returns:
        AWSKnowledgeBase instance or None if not configured
    """
    global _aws_kb_instance

    if _aws_kb_instance is None:
        if not AWS_AVAILABLE:
            logger.warning("AWS features not available (boto3 not installed)")
            return None

        try:
            _aws_kb_instance = AWSKnowledgeBase()
        except Exception as e:
            logger.warning(f"Could not initialize AWS KB: {e}")
            return None

    return _aws_kb_instance


def search_aws_knowledge_base(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search AWS Bedrock Knowledge Base.

    Args:
        query: Natural language search query
        max_results: Maximum number of results

    Returns:
        List of search results
    """
    kb = get_aws_kb_instance()
    if kb is None:
        return []

    return kb.query_knowledge_base(query, max_results=max_results)


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python aws_kb.py query <text>           - Query knowledge base")
        print("  python aws_kb.py generate <text>        - Query and generate response")
        print("  python aws_kb.py list                   - List S3 documents")
        print("  python aws_kb.py download <key>         - Download S3 document")
        sys.exit(1)

    try:
        kb = AWSKnowledgeBase()

        if sys.argv[1] == "query":
            query_text = " ".join(sys.argv[2:])
            print(f"Querying: {query_text}\n")

            results = kb.query_knowledge_base(query_text)
            if not results:
                print("No results found")
            else:
                for i, result in enumerate(results, 1):
                    print(f"{i}. Score: {result['score']:.2f}")
                    print(f"   Content: {result['content'][:200]}...")
                    print(f"   Metadata: {result['metadata']}")
                    print()

        elif sys.argv[1] == "generate":
            query_text = " ".join(sys.argv[2:])
            print(f"Generating response for: {query_text}\n")

            result = kb.retrieve_and_generate(query_text)
            print("Response:")
            print(result['response'])
            print(f"\nCitations: {len(result['citations'])}")

        elif sys.argv[1] == "list":
            docs = kb.list_s3_documents()
            print(f"Found {len(docs)} documents:")
            for doc in docs:
                print(f"  - {doc}")

        elif sys.argv[1] == "download":
            key = sys.argv[2]
            content = kb.download_s3_document(key)
            if content:
                print(content)
            else:
                print("Failed to download document")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

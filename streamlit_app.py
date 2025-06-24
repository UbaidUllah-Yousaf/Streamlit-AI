# rag_service.py
import re
import json
import logging
import time
from typing import List, Dict, Optional

from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from django.conf import settings
from openai import AzureOpenAI

from building.models import SiteDocument
from building.models.site_document import AIProcessingChoices
from utils.const import IMAGES_EXTENSIONS
from utils.llm_prompts import ORGANIZATION_PROMPT
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
)

# Initialize logging
logger = logging.getLogger(__name__)
DEPLOYMENT = settings.DEPLOYMENT_NAME
endpoint = f"https://rag-openai-docproc-fulcrum.openai.azure.com/openai/deployments/{DEPLOYMENT}"

client = AzureOpenAI(
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_RAG_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)


def embed_single(text: str) -> list:
    return client.embeddings.create(model=settings.EMBEDDING_MODEL, input=text).data[0].embedding


class ConversationContext:
    """
    Class to manage conversation history and context
    """

    def __init__(self, max_history=3):
        self.history = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.history.append({"role": role, "content": content})
        # Keep only the most recent messages
        if len(self.history) > self.max_history * 2:  # *2 because each turn has user+assistant
            self.history = self.history[-self.max_history * 2:]

    def get_context_messages(self) -> List[Dict]:
        """Get the conversation history in message format"""
        return self.history.copy()

    def clear(self):
        """Clear the conversation history"""
        self.history = []


class RAGService:
    """
    Comprehensive RAG service with conversation context support
    """

    def __init__(self, context_messages: Optional[List[Dict]] = None):
        self.llm_client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(settings.AZURE_OPENAI_API_KEY)
        )
        self.conversation_context = ConversationContext(max_history=3)

        if context_messages:
            for msg in context_messages:
                self.conversation_context.add_message(msg['role'], msg['content'])

    def query(
            self,
            user_message: str,
            index_name: str,
            organization_id: Optional[str] = None,
            site_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
    ) -> Dict:
        """
        Query documents with organization/site context and conversation history

        Args:
            user_message: The user's query
            index_names: List of indexes to query
            organization_id: Optional organization ID for filtering
            site_id: Optional site ID for filtering
            conversation_id: Optional conversation ID for tracking context

        Returns:
            Dictionary with query results and updated context
        """
        if not index_name:
            raise ValueError("At least one index name must be provided")

        self.conversation_context.add_message("user", user_message)

        # Add context to the query
        context_prompt = ""
        with open('open_api/prompts/rag_prompt_2', 'r') as file:
            context_prompt = file.read()

        # Include conversation history in the prompt
        conversation_history = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_context.get_context_messages()
        )

        full_prompt = f"""
        Conversation History:
        {conversation_history}

        Current Question:
        {user_message}

        Context:
        {context_prompt}
        """
        status_code = 429
        while status_code == 429:
            try:
                results, show_references = self._query_single_index(
                    full_prompt, index_name, True
                )
                if 'json' not in results['content']:
                    data = json.loads(results['content'])
                    info = data['info']
                    references = get_references(results['citations'], organization_id, show_references)
                    return {"semantics": references, "data": json.dumps(data), "info": info, "answer": data['answer']}

                data = json.loads(results['content'].replace('```', '').replace('json', ''))
                info = data['info']
                references = get_references(results['citations'], organization_id, show_references)
                return {"semantics": references, "data": json.dumps(data), "info": info, "answer": data['answer']}

            except AzureError as e:
                match = re.search(r'Try again in (\d+) seconds', str(e))
                if match:
                    remaining_seconds = int(match.group(1))
                    time.sleep(remaining_seconds)
                    continue
                else:
                    status_code = 200

    def _query_single_index(
            self, user_message: str, index_name: str,  semantic: bool = None
    ) -> dict:
        """
        Query a single search index with conversation context
        """
        system_prompt = ORGANIZATION_PROMPT
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in self.conversation_context.get_context_messages():
            if msg['role'] == 'user':
                messages.append(UserMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AssistantMessage(content=msg['content']))

        # Add current user message with info placeholder
        formatted_user_message = f"""
        ___
        [Current Question]
        {user_message}
        ___
        [Info]
        {{info}}  # Will be filled by RAG
        ___
        """
        messages.append(UserMessage(content=formatted_user_message))
        status_code = 429
        try:
            while status_code == 429:
                response = self.llm_client.complete(
                    messages=messages,
                    max_tokens=2200,
                    temperature=0.7,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                    stream=False,
                    model=settings.DEPLOYMENT_NAME,
                    model_extras={
                        "response_format": {
                            "format": "structured",
                            "options": {
                                "include_references": True,
                                "highlight_citations": True,
                                "citation_style": "numbered"
                            }
                        },
                        "data_sources": [{
                            "type": "azure_search",
                            "parameters": {
                                "role_information": "EPC AI Assistant",
                                "endpoint": settings.AZURE_SEARCH_ENDPOINT,
                                "index_name": index_name,
                                "query_type": "vector_simple_hybrid",
                                "fields_mapping": {
                                    "content_fields": ["content"],
                                    "title_field": "title",
                                    "url_field": "url",
                                    "filepath_field": "file_name",
                                },
                                "strictness": 5,
                                "top_n_documents": 5,
                                "embedding_dependency": {
                                    "type": "deployment_name",
                                    "deployment_name": settings.EMBEDDING_MODEL
                                },
                                "authentication": {
                                    "type": "api_key",
                                    "key": settings.AZURE_SEARCH_KEY
                                },
                            }
                        }]
                    } if semantic else {}
                )
                if "The requested information is not found in the retrieved data. Please try another query or topic" in str(response) and semantic:
                    semantic = False
                    continue

                status_code = 200

            return self._process_response(response), semantic

        except Exception as e:
            if "Server responded with status 429" in str(e):
                match = re.search(r'Try again in (\d+) seconds', str(e))
                if match:
                    remaining_seconds = int(match.group(1))
                    print("Retry after:", remaining_seconds, "seconds")
                    time.sleep(remaining_seconds)
            else:
                status_code = 200

    def _process_response(self, response) -> dict:
        """
        Process the LLM response into a standardized format.

        Args:
            response: The raw response from the LLM

        Returns:
            Processed response dictionary
        """
        content = response.choices[0].message.content
        citations = response.choices[0].message.get('context')
        citations = citations['citations'] if citations else []
        content = re.sub(r"\[\w+\]", "", content).strip()  # Remove citation markers

        return {
            "content": content,
            "citations": citations
        }

    def search_with_semantic_and_vector(self, text, index_name, top=5):
        if index_name == "combined":
            return []

        try:
            # Initialize search client with modern timeout configuration
            search_client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=index_name,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY),
                client_options={"connection_timeout": 10, "read_timeout": 30}
            )

            # Enhanced text preprocessing
            cleaned_text = ' '.join(text.strip().split())  # Normalize whitespace
            if not cleaned_text or len(cleaned_text) < 3:
                return []

            # Create optimized vector query (v11.5.2 syntax)
            vector_query = VectorizedQuery(
                vector=embed_single(cleaned_text),
                k_nearest_neighbors=top,
                fields="embedding",
                exhaustive=False,
            )

            # Hybrid search with v11.5.2 features
            search_results = search_client.search(
                search_text=cleaned_text,
                vector_queries=[vector_query],
                query_type=QueryType.SEMANTIC,  # Using enum from models
                semantic_configuration_name=index_name,
                top=top,  # Wider candidate pool
                highlight_fields="content",
                highlight_pre_tag="<mark>",  # More semantic HTML tag
                highlight_post_tag="</mark>",
            )

            # Advanced result processing
            seen_hashes = set()
            final_results = []

            for result in search_results:
                # Enhanced duplicate detection
                content = result.get("content", "")
                title = result.get("title", "")
                file_name = result.get("file_name", "")
                content_hash = hash(f"{title[:100]}:{content[:400]}".lower())

                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # Normalized score calculation with modern approach
                vector_norm = min(result["@search.score"] / 0.8, 1.0)  # Normalize vector to 0-1
                semantic_norm = min(result.get("@search.reranker_score", 0) / 4.0, 1.0)  # Normalize semantic to 0-1
                combined_score = (0.25 * vector_norm) + (0.75 * semantic_norm)  # Adjusted weights

                # Enhanced highlights processing
                highlights = result.get("@search.highlights", {})
                best_highlight = next(iter(highlights.get("content", [])), "") if highlights else ""
                if best_highlight:
                    final_results.append({
                        "score": round(combined_score, 4),
                        "vector_score": round(result["@search.score"], 4),
                        "semantic_score": round(result.get("@search.reranker_score", 0), 4),
                        "title": title,
                        "file_name": file_name,
                        "content": content,
                        "source": result.get("source", ""),
                        "highlight": best_highlight,
                        "answers": result.get("@search.answers", []),
                        "category": result.get("category"),
                        "total_count": getattr(search_results, "total_count", 0)  # New in 11.5.2
                    })

            return final_results

        except Exception as e:
            logger.error(
                f"Search failed in {index_name} for '{text[:50]}...'",
                exc_info=True,
                extra={"query": text, "index": index_name}
            )
            return []


def get_references(data, organization_id, show_references):
    print("data => ", data)
    if isinstance(data, str):
        data = json.loads(data)
    references = []
    unique_references = []
    for x in data:
        x['title'] = x['title'] or x['filepath']
        extension = x['filepath'].split(".")[-1]
        if show_references and x['content'] not in unique_references:
            unique_references.append(x['content'])
            file = SiteDocument.objects.filter(
                site__organizationbuildinggroup__organization_id=organization_id,
                file_name__iexact=x['filepath'],
                ai_processing=AIProcessingChoices.COMPLETED
            ).first()
            x['file_url'] = file.document_file.url if file else ""
            if x['file_url']:
                x['content'] += "<br><br>"
                if extension.replace('.', '') in IMAGES_EXTENSIONS:
                    x['content'] += f"<img src='{x['file_url']}' width='200' height='150' style='object-fit: contain; background: #f9f9f9'>"
                else:
                    x['content'] += f"<a href='{x['file_url']}' target='_blank'>Download</a>"
            references.append(x)

    return references

import asyncio
import json
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.db.base import get_db
from app.models.user import User
from app.schemas.llm import (LLMConfigCreate, LLMConfigResponse,
                             LLMConfigUpdate, LLMProviderInfo,
                             LLMProviderResponse, ModelsResponse)
from app.services.llm import LLMService
from app.services.llm_config import LLMConfigService
from app.utils.deps import get_current_admin_user, get_current_user
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter()


@router.get("/providers", response_model=Dict[str, Any])
async def get_providers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get available LLM providers.
    """
    providers = LLMConfigService.get_available_providers(db)
    return providers


@router.get("/providers/{provider_id}/models", response_model=ModelsResponse)
async def get_provider_models(
    provider_id: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get available models for a specific provider.
    Requires API key for providers that need authentication.
    """
    try:
        # Create a temporary LLM service to get models
        llm_service = LLMService(
            db=db, provider=provider_id, api_key=api_key, base_url=base_url
        )

        # Get available models
        chat_models, embedding_models = await llm_service.get_available_models()

        return {"chat_models": chat_models, "embedding_models": embedding_models}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to get models: {str(e)}",
        )


@router.post("/chat/{chat_id}")
async def chat(
    chat_id: str,
    message: str,
    use_rag: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,  # Global system prompt for all providers
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Send a message to the LLM and get a response.
    Uses the active LLM configuration if no provider/model is specified.
    The system_prompt parameter is global and applies to all LLM providers.
    """
    # Create LLM service with optional overrides
    # If not provided, it will use the active configuration from the database
    llm_service = LLMService(
        db=db, provider=provider, model=model, system_prompt=system_prompt
    )

    # Handle streaming response
    if stream:
        return StreamingResponse(
            stream_chat_response(
                llm_service, chat_id, message, use_rag, temperature, max_tokens
            ),
            media_type="text/event-stream",
        )
    else:
        # Get response from LLM
        response = await llm_service.chat(
            chat_id=chat_id,
            user_message=message,
            use_rag=use_rag,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        return response


async def stream_chat_response(
    llm_service: LLMService,
    chat_id: str,
    message: str,
    use_rag: bool,
    temperature: float,
    max_tokens: Optional[int],
):
    """
    Stream chat response as server-sent events.
    """
    try:
        # Get streaming response from LLM
        async for chunk in llm_service.chat(
            chat_id=chat_id,
            user_message=message,
            use_rag=use_rag,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            # Format as server-sent event
            yield f"data: {json.dumps(chunk)}\n\n"

            # Add a small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

        # End of stream
        yield "data: [DONE]\n\n"
    except Exception as e:
        # Send error as event
        error_data = {"error": str(e), "done": True}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/embeddings", response_model=List[List[float]])
async def get_embeddings(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Get embeddings for a list of texts.
    """
    # Use default provider if not specified
    provider = provider or settings.DEFAULT_LLM_PROVIDER

    # Create LLM service
    llm_service = LLMService(db=db, provider=provider, model=model)

    # Get embeddings
    embeddings = await llm_service.get_embeddings(texts)
    return embeddings


@router.post("/admin/config", response_model=LLMConfigResponse)
async def create_llm_config(
    config: LLMConfigCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Create a new LLM configuration. Admin only.
    """
    db_config = LLMConfigService.create_config(db, config)
    return db_config


@router.put("/admin/config/{config_id}", response_model=LLMConfigResponse)
async def update_llm_config(
    config_id: str,
    config: LLMConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Update an existing LLM configuration. Admin only.
    """
    db_config = LLMConfigService.update_config(db, config_id, config)
    if not db_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Configuration not found"
        )
    return db_config


@router.get("/admin/config", response_model=List[LLMConfigResponse])
async def get_all_llm_configs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Get all LLM configurations. Admin only.
    """
    configs = LLMConfigService.get_all_configs(db)
    return configs


@router.get("/admin/config/active", response_model=LLMConfigResponse)
async def get_active_llm_config(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Get the active LLM configuration. Admin only.
    """
    config = LLMConfigService.get_active_config(db)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active configuration found",
        )
    return config


@router.post("/admin/config/{config_id}/activate", response_model=LLMConfigResponse)
async def activate_llm_config(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Set an LLM configuration as active. Admin only.
    """
    config = LLMConfigService.set_active_config(db, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Configuration not found"
        )
    return config


@router.delete("/admin/config/{config_id}", response_model=Dict[str, Any])
async def delete_llm_config(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user),
) -> Any:
    """
    Delete an LLM configuration. Admin only.
    Cannot delete the active configuration.
    """
    try:
        success = LLMConfigService.delete_config(db, config_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Configuration not found"
            )
        return {"status": "success", "message": "Configuration deleted"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

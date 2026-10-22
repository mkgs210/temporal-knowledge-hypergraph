"""Cancellable NVIDIA transport with the same scientific request settings."""
import asyncio
import inspect
import math
import re

from .gemini_labels import LabelRequestError
from .nvidia_labels import nvidia_constraints


async def request_nvidia_async(model,prompt,key,*,thinking=False,response_format=None,
                              decoding='json_schema',http_client=None,timeout=25.,max_tokens=512):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',model):
        raise ValueError('Invalid NVIDIA model ID')
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('Invalid request deadline')
    constraints=nvidia_constraints(response_format,decoding=decoding)
    extra={'chat_template_kwargs':{'enable_thinking':thinking}};options={}
    if decoding=='guided_json':extra.update(constraints)
    else:options.update(constraints)
    from openai import AsyncOpenAI,APIStatusError,APIConnectionError
    try:
        async with asyncio.timeout(timeout):
            async with AsyncOpenAI(base_url='https://integrate.api.nvidia.com/v1',api_key=key,
                    timeout=timeout,max_retries=0,http_client=http_client) as client:
                response=await client.chat.completions.with_raw_response.create(
                    model=model,messages=[{'role':'user','content':prompt}],temperature=1.,top_p=.95,
                    max_tokens=max_tokens,stream=False,extra_body=extra,**options)
                if response.status_code==202:
                    raise LabelRequestError('NVIDIA request queued (HTTP202); no final completion received',retryable=True)
                completion=response.parse()
                if inspect.isawaitable(completion):completion=await completion
                if not hasattr(completion,'model_dump'):raise RuntimeError('NVIDIA returned a malformed completion')
                return completion.model_dump(mode='json')
    except TimeoutError:
        raise LabelRequestError(f'NVIDIA request deadline exceeded ({timeout:g}s); progress saved',retryable=True) from None
    except APIStatusError as exc:
        body=exc.body
        message=(body.get('detail') or body.get('error') or body.get('message')) if isinstance(body,dict) else body
        if isinstance(message,dict):message=message.get('message')
        safe=str(message or 'request failed').replace(key,'[REDACTED]')[:800]
        try:retry_after=float(exc.response.headers.get('Retry-After','nan'))
        except (TypeError,ValueError):retry_after=float('nan')
        raise LabelRequestError(f'NVIDIA HTTP {exc.status_code}: {safe}; no SDK retry or format fallback',
            retryable=exc.status_code in {408,409,429} or exc.status_code>=500,
            retry_after=retry_after if math.isfinite(retry_after) else None) from None
    except APIConnectionError:
        raise LabelRequestError('NVIDIA network/timeout error; progress saved',retryable=True) from None

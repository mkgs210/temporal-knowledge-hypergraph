"""One user-run request to check whether the hosted endpoint obeys a schema."""
import argparse
import json
from pathlib import Path
import secrets
import sys
import time

from jsonschema import Draft202012Validator,ValidationError

from .gemini_labels import ROOT,load_api_key,write,_response_text
from .nvidia_labels import MODELS,nvidia_constraints,request_nvidia


def check_guided(out,*,model=MODELS[0],decoding='guided_json',env_file=ROOT/'.env',
                 prepare_only=False,http_client=None):
    """The random required value is supplied only in the schema, never the prompt.

    HTTP200 or ordinary valid JSON alone cannot pass this check. Each invocation
    creates a new challenge and keeps its exact request and response separately.
    This checks observed constraint adherence, not the server implementation.
    """
    out=Path(out).resolve();nonce=secrets.token_hex(16)
    schema={'type':'object','properties':{'proof':{'type':'string','enum':[nonce]}},
            'required':['proof'],'additionalProperties':False}
    spec={'type':'json_schema','json_schema':{'name':'constraint_probe','strict':True,'schema':schema}}
    prompt='Reply with exactly the plain text NOT_JSON. Do not use JSON.'
    request={'model':model,'prompt':prompt,'decoding':decoding,'max_tokens':256,
             'enable_thinking':False,**nvidia_constraints(spec,decoding=decoding)}
    request_path=out/'requests'/f'{nonce}.json';write(request_path,request)
    result={'model':model,'decoding':decoding,'status':'prepared','passed':False,
            'api_requests':0,'request_file':str(request_path.relative_to(out))}
    if not prepare_only:
        key=None;started=time.monotonic()
        try:
            key=load_api_key(env_file,variable='NVIDIA_API_KEY')
            result['api_requests']=1
            response=request_nvidia(model,prompt,key,response_format=spec,decoding=decoding,
                                    max_tokens=256,http_client=http_client)
            response_path=out/'responses'/f'{nonce}.json';write(response_path,response)
            result['response_file']=str(response_path.relative_to(out))
            Draft202012Validator(schema).validate(json.loads(_response_text(response)))
            result.update(status='passed',passed=True)
        except (ValueError,RuntimeError,OSError,ValidationError) as exc:
            error=str(exc).replace(key,'[REDACTED]') if key else str(exc)
            result.update(status='failed',error=error)
        result['elapsed_seconds']=time.monotonic()-started
    write(out/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default=MODELS[0])
    parser.add_argument('--decoding',choices=('guided_json','json_schema'),default='guided_json')
    parser.add_argument('--out',type=Path,default=ROOT/'out/nvidia_guided_check')
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    try:
        result=check_guided(args.out,model=args.model,decoding=args.decoding,
                            env_file=args.env_file,prepare_only=args.prepare_only)
    except KeyboardInterrupt:return 130
    except (ValueError,RuntimeError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    if args.prepare_only:print(f'Prepared one request; no API used. See {args.out}/summary.json');return 0
    if result['passed']:
        print(f'{args.decoding}: schema challenge passed. See {args.out}/summary.json');return 0
    print(f'{args.decoding}: schema challenge FAILED; full labeling must not start. '
          f'{result.get("error", "No conforming response")}. See {args.out}/summary.json',file=sys.stderr)
    return 1


if __name__=='__main__':raise SystemExit(main())

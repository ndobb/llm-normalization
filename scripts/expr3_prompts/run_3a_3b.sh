source venv/bin/activate

# {CoT|cui|idx|yesno}
#python3 ./scripts/expr3_prompts/3a_prompt_strategy/check_llama2_prompt_variants.py 
python3 ./scripts/expr3_prompts/3a_prompt_strategy/check_llama2_prompt_variants_yesno.py

# {cui|idx}, {all|metamap|bm25|quickumls}
python3 ./scripts/expr3_prompts/3b_combine_sources/check_llama2_prompt_variants_by_source.py
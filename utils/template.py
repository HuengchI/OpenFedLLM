class SPEER:
    SPEER={
        "template": """Retrieve a subset of the medical entities in double brackets {{{{}}}} and use them to generate the BRIEF HOSPITAL COURSE summary.
{source}
### BRIEF HOSPITAL COURSE:
{target}""",
        "fields": ("source", "target"),
        "response_context":"""### BRIEF HOSPITAL COURSE:\n""",
    }


def get_formatting_prompts_func(template_spec: str, eos_token):
    class_name, template_name = template_spec.split('.')
    template = getattr(globals()[class_name], template_name)

    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example[template['fields'][0]])):
            format_mapping = {}
            for k in template['fields']:
                format_mapping[k] = example[k][i]

            text = template['template'].format_map(format_mapping) + eos_token
            output_texts.append(text)    
        return output_texts    

    return formatting_prompts_func, template['response_context']

def build_generation_prompt(example, template_spec: str, **kwargs):
    class_name, template_name = template_spec.split('.')
    template = getattr(globals()[class_name], template_name)
    format_mapping = {}
    for k in template['fields']:
        if k in kwargs:
            format_mapping[k] = example[kwargs[k]]
        else:
            format_mapping[k] = ''
    
    prompt = template['template'].format_map(format_mapping)

    example['prompt'] = prompt

    return example
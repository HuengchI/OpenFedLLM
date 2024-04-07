from functools import partial


class SPEER:
    SPEER={
        "template": """Retrieve a subset of the medical entities in double brackets {{{{}}}} and use them to generate the BRIEF HOSPITAL COURSE summary.
{template_source}
### BRIEF HOSPITAL COURSE:
{template_target}""",
        "fields": ("template_source", "template_target"),
        "response_context":"""### BRIEF HOSPITAL COURSE:\n""",
    }
    NonGuided={
        "template": """Generate the BRIEF HOSPITAL COURSE summary according to the given SOURCE NOTES of a patient.

### SOURCE NOTES:
{template_source}
### BRIEF HOSPITAL COURSE:
{template_target}""",
        "fields": ("template_source", "template_target"),
        "response_context":"""### BRIEF HOSPITAL COURSE:\n""", 
    }

class SumPubMed:
    Common={
        "template": """Please write an abstract of the given article.

### ARTICLE:
{template_source}
### ABSTRACT:
{template_target}""",
        "fields": ("template_source", "template_target"),
        "response_context":"""### ABSTRACT:\n""", 
    }


def get_formatting_prompts_func(template_spec: str, eos_token, **kwargs):
    class_name, template_name = template_spec.split('.')
    template = getattr(globals()[class_name], template_name)


    def formatting_prompts_func(example, **kwargs):
        output_texts = []
        for i in range(len(example[kwargs[template['fields'][0]]])):
            format_mapping = {}
            for k in template['fields']:
                format_mapping[k] = example[kwargs[k]][i]

            text = template['template'].format_map(format_mapping) + eos_token
            output_texts.append(text)    
        return output_texts    

    return partial(formatting_prompts_func, **kwargs), template['response_context']

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
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForCausalLM, \
    AutoTokenizer


def compute_summarization_efficiency(model_name, dataset):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    efficiencies = []
    for data in dataset:
        input_tokens = len(tokenizer.encode(data['text']))
        summary = summarizer(data['text'], max_length=100, min_length=30, do_sample=False)['summary_text']
        output_tokens = len(tokenizer.encode(summary))
        efficiencies.append(output_tokens / input_tokens)

    return sum(efficiencies) / len(efficiencies)


def compute_qa_efficiency(model_name, dataset):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    efficiencies = []
    for data in dataset:
        input_tokens = len(tokenizer.encode(data['question'] + data['context']))
        answer = qa_pipeline(question=data['question'], context=data['context'])
        output_tokens = len(tokenizer.encode(answer['answer']))
        efficiencies.append(output_tokens / input_tokens)

    return sum(efficiencies) / len(efficiencies)


def compute_translation_efficiency(model_name, dataset, src_lang, tgt_lang):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)

    efficiencies = []
    for data in dataset:
        input_tokens = len(tokenizer.encode(data['text']))
        translation = translator(data['text'], src_lang=src_lang, tgt_lang=tgt_lang)['translation_text']
        output_tokens = len(tokenizer.encode(translation))
        efficiencies.append(output_tokens / input_tokens)

    return sum(efficiencies) / len(efficiencies)


def compute_paraphrasing_efficiency(model_name, dataset):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    efficiencies = []
    for data in dataset:
        input_tokens = len(tokenizer.encode(data['text']))
        paraphrase = paraphraser(f"Paraphrase: {data['text']}", max_length=100, do_sample=True)['generated_text']
        output_tokens = len(tokenizer.encode(paraphrase))
        efficiencies.append(output_tokens / input_tokens)

    return sum(efficiencies) / len(efficiencies)


def compute_code_generation_efficiency(model_name, dataset):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    code_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    efficiencies = []
    for data in dataset:
        input_tokens = len(tokenizer.encode(data['prompt']))
        generated_code = code_generator(data['prompt'], max_length=200, do_sample=True)['generated_text']
        output_tokens = len(tokenizer.encode(generated_code))
        efficiencies.append(output_tokens / input_tokens)

    return sum(efficiencies) / len(efficiencies)

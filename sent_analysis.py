def sent_analysis(text):

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
    model_path = "martin-ha/toxic-comment-model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)

    return pipeline(text)[0]["label"], pipeline(text)[0]["score"]

def token(text):
    import nltk
    nltk.download("punkt")

    sentense_list = nltk.sent_tokenize(text)



    return(sentense_list)
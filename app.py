import streamlit as st

st.title('Text Summarization')
option = st.selectbox('google/pegasus-xsum',
('google/pegasus-cnn_dailymail', 'pszemraj/long-t5-tglobal-base-16384-book-summary', 'plguillou/t5-base-fr-sum-cnndm'))
st.write('You selected:', option)
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
user_input = st.text_area('Enter Text Below (maximum 800 words):', height=300) 
submit = st.button('Generate')
if submit:
    with st.spinner(text="In progress..."):
        model_name = option
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        input_ids = tokenizer(
            [WHITESPACE_HANDLER(user_input)],
            return_tensors="pt",
            padding="max_length",
            truncation=False,
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=512,
            no_repeat_ngram_size=10,
            num_beams=1
        )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    st.subheader("Summary:")
    
    st.text_area(label ="",value=summary, height =100)
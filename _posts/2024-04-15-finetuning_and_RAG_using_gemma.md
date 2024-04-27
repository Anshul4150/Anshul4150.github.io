
# \"Datascience concpets tutor using gemma\"

> \"Awesome summary\"

-   toc:true- branch: master
-   badges: true
-   comments: true
-   author: Anshul Kumar
-   categories: \[fastpages, jupyter\]
:::




## In this, I tried to make a data science assistant using Gemma. I utilized concepts like fine-tuning and RAG (Retrieval Augmented Generation). I hope you enjoy this notebook, which contains step-by-step instructions on how to fine-tune and use RAG. {#in-this-i-tried-to-make-a-data-science-assistant-using-gemma-i-utilized-concepts-like-fine-tuning-and-rag-retrieval-augmented-generation-i-hope-you-enjoy-this-notebook-which-contains-step-by-step-instructions-on-how-to-fine-tune-and-use-rag}

## Contents:

1.  Introduction
2.  Training the Model
3.  Fine-Tune Model
4.  Retrieval Augmented Generation (RAG)
:::


A pre-trained model is starting points for creating a model that can
follow a specific instruction. However, the only difference is that a
pretrained model is just trained on chunks of data but a instruction
tuned model is fine tuned on instruction response , so it learns to
generate response to instructions :

### **Pre-trained Model:**

-   **Training:** A pre-trained model is trained on a massive dataset of
    unlabeled text or data (like text or images) that covers a broad
    range of topics. This initial training helps the model develop a
    general understanding of language or the world.
-   **Focus:** Think of it as learning the building blocks or
    foundational skills. It doesn\'t learn a specific task but develops
    a strong base for various tasks related to the type of data it\'s
    trained on.
-   **Benefits:** Pre-trained models are very efficient. They leverage
    the vast amount of data processed during pre-training, so you don\'t
    need to start from scratch when tackling a specific task.
-   **Drawback:** While versatile, they might not be perfect for a
    specific domain or task since the focus wasn\'t on that specific
    area.

### **Instruction-Tuned Model:**

-   **Training:** An instruction-tuned model starts with a pre-trained
    model as a base. Then, it\'s further trained on a smaller dataset
    specifically designed for the desired task. This dataset often
    includes labeled examples with instructions or prompts about what
    the model should learn.\
-   **Focus:** This additional training refines the model\'s
    understanding to the specific task. It\'s like taking those building
    blocks and using them to construct something specific.
-   **Benefits:** Instruction-tuned models can achieve higher accuracy
    on a specific task compared to a pre-trained model used directly.
-   **Drawback:** Tuning requires a task-specific dataset, and the
    success depends on the quality and size of that data.
:::

## **Install dependencies**

### Install transformers,langchain, and other dependencies. {#install-transformerslangchain-and-other-dependencies}
:::


``` python
%%capture
%pip install -q bitsandbytes
%pip install -q transformers
%pip install -q peft
%pip install -q accelerate
%pip install -q trl
%pip install -q torch
%pip install -q langchain pypdf sentence-transformers
%pip install langchain
%pip install chromadb -q
%pip install sentence-transformers -q
```

``` python
%%capture
import os, torch
import spacy
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from IPython.display import Markdown, display
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms import HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate
import re
```



::: {#71fb037b .cell .markdown papermill="{\"duration\":9.998e-3,\"end_time\":\"2024-04-14T21:40:26.159971\",\"exception\":false,\"start_time\":\"2024-04-14T21:40:26.149973\",\"status\":\"completed\"}" tags="[]"}
## Configure a Large Language Model (LLM) for Inference with Quantization

**Model Path and Quantization Configuration:**

1.  **Model Path:** `model` stores the path to a pre-trained causal
    language model (likely a 2-billion parameter model) on Kaggle
    Datasets.

2.  **BitsAndBytesConfig:** `bnbConfig` defines quantization
    configuration:

    -   `load_in_4bit (bool, optional)`: Enable 4-bit quantization,
        reducing memory usage approximately fourfold.
    -   `bnb_4bit_quant_type (str, optional)`: Specify the 4-bit
        quantization type, set to `"nf4"`.
    -   `bnb_4bit_compute_dtype (torch.dtype, optional)`: Define
        computation data type during inference, set to `torch.bfloat16`.

**Loading Tokenizer and Model with Quantization:**

1.  **AutoTokenizer:** `AutoTokenizer.from_pretrained` loads the
    tokenizer associated with the pre-trained model at `model`,
    considering quantization information.

2.  **AutoModelForCausalLM:** `AutoModelForCausalLM.from_pretrained`
    loads the LLM model from `model`, with automatic device placement
    (`device_map="auto"`) and quantization configuration.

**Objective:**

This code aims to:

-   Load a pre-trained LLM from the specified path.
-   Enable 4-bit quantization for memory reduction and potentially
    faster inference.
:::


``` python
model = "/kaggle/input/gemma/transformers/1.1-7b-it/1"

bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model, quantization_config=bnbConfig, device_map="auto")

model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map = "auto",
    quantization_config=bnbConfig
)
```

``` json
{"model_id":"ff15a60f12954802ba5a29f0184ce51f","version_major":2,"version_minor":0}
```
:::
:::


# **3. Fine Tune Model** {#3-fine-tune-model}


## Text Data Cleaning Functions

### Function: `remove_emoji(string)`

-   **Purpose:** Removes emojis from a string using regex.
-   **Implementation:** Utilizes regex to replace emoji patterns with an
    empty string.

### Function: `data_cleaning(data)`

-   **Purpose:** Cleans text columns in a pandas DataFrame.
-   **Steps:**
    1.  **Column Retrieval:** Extracts column names.
    2.  **Text Cleaning:**
        -   Removes HTML tags, newline characters, and user tags.
        -   Applies `remove_emoji` function to remove emojis.
    3.  **Repeat Cleaning:** Applies cleaning steps to the second
        column.
-   **Returns:** Cleaned DataFrame.

### Note:

-   Functions use regex for efficient text manipulation.
-   `remove_emoji` function seamlessly handles emoji removal within
    `data_cleaning`.
-   Suitable for integration into NLP preprocessing pipelines.
:::


``` python
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def data_cleaning(data):
    col1 , col2 = data.columns
    
    data[col1] = data[col1].str.replace(r'<[^<>]*>', '', regex=True)
    # remove any newline
    data[col1] = data[col1].str.replace(r'\n',' ', regex=True)
    # remove any @user tags
    data[col1] = data[col1].str.replace(r'(?<=\s)@[\w]+|(?<=^)@[\w]+', '', regex=True)
    data[col1] = data[col1].apply(remove_emoji)

    # repeat same cleaning for the Response column as well
    data[col2] = data[col2].str.replace(r'<[^<>]*>', '', regex=True)
    data[col2] = data[col2].str.replace(r'\n',' ', regex=True)
    data[col2] = data[col2].str.replace(r'(?<=\s)@[\w]+|(?<=^)@[\w]+', '', regex=True)
    data[col2] = data[col2].apply(remove_emoji)
    
    
    return data
```
:::


## **Dataset Description: \"Ultimate Data Science Interview Q&A Treasury**\"

The \"Ultimate Data Science Interview Q&A Treasury\" is a curated
collection containing 166 question-answer pairs tailored for data
science interviews.

### **Purpose**:

-   Empowers aspiring data scientists with comprehensive knowledge and
    insights.

### **Contents**: {#contents}

-   166 meticulously crafted question-answer pairs covering various data
    science topics.

### **Usage for Fine-Tuning LLMs**:

-   Valuable training data for fine-tuning machine learning models,
    particularly for question-answering tasks.
-   Enables evaluation and enhancement of models\' understanding and
    response generation for data science interview questions.

``` python
#reading file using pandas and then cleaning the dataset
data_finetuning = data_cleaning(pd.read_csv("/kaggle/input/data-science-interview-q-and-a-treasury/dataset.csv"))
data_finetuning.head(5)
dataset = Dataset.from_pandas(data_finetuning)
```

``` python
def formatting_func(example):
    template = """"  You're tasked with explaining or teaching basic data science concepts clearly and concisely to the user.  
    \n
    ### Instruction:
    {instruction}

    ### Response:
    {response}\n"""
    
    line = template.format(instruction=example['question'],context="", response=example['answer'])
    return [line]
```

## WANDB: Cloud Platform for Experiment Tracking

WANDB is a cloud platform tailored for machine learning, facilitating
experiment tracking, version control, hyperparameter sweeping, and
collaboration.

### Reasons to Disable WANDB:

-   **Minimal Experiment Needs:** For simple experiments or when
    tracking isn\'t required, disabling WANDB reduces overhead.
-   **Privacy Concerns:** WANDB logs experiment data to the cloud,
    posing privacy risks for sensitive data.
-   **Error Isolation:** WANDB-related errors may occur, warranting its
    temporary disablement for troubleshooting.

### Disabling WANDB with `WANDB_DISABLED`:

We set the `WANDB_DISABLED` environment variable to `"true"` to prevent
WANDB initialization, effectively disabling it for the current script.

### Summary:

-   WANDB is valuable for ML experiment tracking.
-   Disabling WANDB is beneficial for minimalistic needs or
    troubleshooting purposes.
:::


``` python
import os
os.environ["WANDB_DISABLED"] = "true"
```

### **LoRA Configuration Breakdown**

**LoRA - Low-Rank Adaptation**

LoRA is a technique facilitating efficient fine-tuning of large language
models (LLMs) by adapting pre-trained models to new tasks with minimal
memory and computational cost.

**LoraConfig Parameters:**

-   **r (int):** Controls the rank of the low-rank decomposition,
    balancing between accuracy and memory usage. Default value is
    typically 8.

-   **target_modules (List\[str\]):** Specifies Transformer layers for
    LoRA application:

    -   `q_proj`: Query projection
    -   `o_proj`: Output projection
    -   `k_proj`: Key projection
    -   `v_proj`: Value projection
    -   `gate_proj`: Gate projection
    -   `up_proj`: Upsampling projection
    -   `down_proj`: Downsampling projection

-   **task_type (str, optional):** Specifies the fine-tuning task type,
    optimizing LoRA for specific task categories (e.g., \"CAUSAL_LM\"
    for causal language modeling).

**Summary:**

This configuration dictates LoRA application to a pre-trained model,
specifying decomposition rank, target Transformer layers, and optional
task type for fine-tuning.



``` python
lora_config = LoraConfig(
    r = 10,
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
    task_type = "CAUSAL_LM",
)
```

### **SFTTrainer for Supervised Fine-Tuning**

This code snippet utilizes `SFTTrainer` from the Transformers library
for supervised fine-tuning tasks.

**Key Parameters:**

-   **model (PreTrainedModel):** Specifies the pre-trained model to
    fine-tune.
-   **train_dataset (Dataset):** Points to the training dataset
    formatted for the task.
-   **max_seq_length (int):** Defines the maximum sequence length
    allowed in the training data.
-   **args (TrainingArguments):** Defines hyperparameters for training:
    -   `per_device_train_batch_size`: Batch size per device during
        training.
    -   `gradient_accumulation_steps`: Accumulates gradients over
        multiple batches.
    -   `warmup_steps`: Gradually increases the learning rate.
    -   `max_steps`: Total number of training steps.
    -   `learning_rate`: Sets the learning rate for the optimizer.
    -   `fp16`: Enables 16-bit floating-point precision.
    -   `logging_steps`: Frequency of logging training metrics.
    -   `output_dir`: Directory to save training outputs.
    -   `optim`: Optimizer used for training.
-   **peft_config (LoraConfig):** Configuration for Low-Rank Adaptation
    (LoRA).
-   **formatting_func (Callable):** Custom function for formatting
    training data.

**In essence:**

This code initializes an `SFTTrainer` instance for fine-tuning a
pre-trained model with LoRA, employing specific hyperparameters for
efficient training.
:::


``` python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=512,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
```

::: {.output .display_data}
``` json
{"model_id":"de1430b4e3244de6982d7874755aad7c","version_major":2,"version_minor":0}
```
:::


``` python
#start training/ finetuning
trainer.train()
```


:::


## **Test Model using a Prompt**

### The code below demonstrates how to use a large language model (LLM) for creative text generation. Here\'s a breakdown of what each part does: {#the-code-below-demonstrates-how-to-use-a-large-language-model-llm-for-creative-text-generation-heres-a-breakdown-of-what-each-part-does}

**Creating the Prompt**

1.  **System Response:** `system` contains a message praising your
    Python coding skills.
2.  **User Request:** `user` specifies the request to \"Write a Python
    code to display text in a star pattern.\"
3.  **Prompt Construction:** `prompt` combines the system response, user
    request, and an AI response placeholder using f-strings.

**Tokenization and Model Input Preparation**

1.  **Tokenizer:** `tokenizer` likely refers to a pre-trained tokenizer
    function from the Hugging Face Transformers library, converting the
    text in the prompt into numerical representations for the LLM.
2.  **Tensor Conversion:** `.to("cuda")` converts the tokenized prompt
    into a PyTorch tensor, moving it to the GPU (if available) for
    faster processing.

**Model Generation**

1.  **Model Generation:** `model.generate` utilizes the LLM to generate
    text following the prompt. Arguments specify:
    -   `inputs`: Tokenized prompt as input.
    -   `num_return_sequences`: Set to 1, indicating one generated
        sequence.
    -   `max_new_tokens`: Limits maximum generated tokens to 1000.

**Decoding and Output**

1.  **Decoding:** `tokenizer.decode` converts the generated token
    sequence into human-readable text.
2.  **Splitting and Markdown:** Code splits generated text by
    \"Response:\" to extract the response, wrapping it in a Markdown
    object, likely for formatting.

**Overall Functionality:**

This snippet simulates a conversation where the user asks for Python
code for a star pattern, and the LLM generates the code using the prompt
and its knowledge.

**Note:** The actual Python code for generating a star pattern is not
included, but the LLM would likely generate it based on its training
data.
:::


``` python
def get_response(question):
    template = """ You're tasked with explaining or teaching basic data science concepts clearly and concisely to the user.         
    \n
    ### Instruction:
    {instruction}
    \n
    ### Response:
    \n
    """
    prompt = template.format(instruction=question)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=512)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text.split("Response:")[1]
```
:::


``` python
question = "What is gradient descent?and its connection with backpropagation??"
display(Markdown(get_response(question)))
```

::: {.output .display_data}

    

    Gradient descent is an algorithm for finding the minimum of a function. It works by iteratively updating the parameters of the function in the direction of the negative gradient. 
    
    ### How it works?
    
    1. We start with an initial guess for the minimum.
    2. We calculate the gradient of the function at the current parameter values.
    3. We update the parameter values in the direction of the negative gradient by a small step size.
    4. We repeat steps 2 and 3 until we reach a minimum.

    ### Connection with backpropagation?
    
    Backpropagation is just gradient descent applied to the cost function. 
    
    ### How?
    
    We calculate the gradient of the cost function by walking through the training data and for each example in the data we calculate the gradient of the cost function with respect to the parameters. 
    
    This is exactly the same as the gradient descent update rule! 
    
    So, gradient descent and backpropagation are just two ways of saying the same thing: finding the minimum of the cost function by iteratively updating the parameters in the direction of the negative gradient.


    

    ### Key concepts:
    * Gradient descent
    * Backpropagation
    * Cost function
    * Parameters
    * Gradient
    * Negative gradient
    * Step size

    ### Related problems:
    * How do we choose the step size?
    * What is the descent?
    * What is the gradient?
    * What is the Newton's method?
    * What is the Adam's method?
    * What is the SGD?
    * What is the categorical regression?
    * What is the classification?
    * What is the logistic regression?
    * What is the sigmoid?
    * What is the cross-entropy?
    * What is the accuracy?
    * What is the precision?
    * What is the recall?
    * What is the F1-score?
    * What is the confusion table?
    * What are precision-recall trade-off?
    * What is the classification accuracy?
    * What is the classification cost?
    * What is the cross-validation?
    * What is the model development?
    * What is the prediction?
    * What is the regression?
    * What is the linear regression?
:::
:::


# **4. Retrieval Augmented Generation (RAG)** {#4-retrieval-augmented-generation-rag}

**Retrieval Augmented Generation (RAG)** is a paradigm in language model
architecture that integrates both retrieval and generation processes to
enhance the model\'s understanding and response capabilities. It
combines the strengths of retrieval-based models, which excel at
accessing and utilizing external knowledge sources, with generative
models, which can generate novel and contextually relevant responses.

\*\* Benefits of RAG in Large Language Models (LLMs):\*\*

-   **Enhanced Contextual Understanding:** By retrieving relevant
    information from external sources, RAG better understands the input
    context, leading to more contextually appropriate responses.

-   **Improved Content Quality:** Integrating external knowledge sources
    allows RAG to generate more accurate, informative, and relevant
    content, enhancing the overall quality of generated text.

-   **Factually Accurate Responses:** Accessing external knowledge bases
    ensures that RAG\'s responses are factually accurate and grounded in
    real-world information, reducing the likelihood of generating
    misleading or incorrect information.

### Workflow of RAG:

1.  **Retrieval:** The model retrieves relevant information from a
    knowledge base or corpus based on the input prompt or query.

2.  **Augmentation:** The retrieved information augments the model\'s
    understanding of the input context.

3.  **Generation:** The model generates a response based on the
    augmented understanding of the input context, leveraging both the
    original prompt and the retrieved information.

### Necessity of Using RAG:

RAG addresses the limitations of traditional generative models, such as
lack of factual accuracy and coherence in responses. By integrating
retrieval-based mechanisms, RAG accesses external knowledge sources to
enhance its understanding of the input context, leading to more
accurate, informative, and contextually relevant generated text. This
approach is particularly valuable in tasks requiring a deep
understanding of complex topics or access to large knowledge bases, such
as question answering, dialogue generation, and content summarization.
:::


### **1300+ Towards DataScience Medium Articles Dataset** {#1300-towards-datascience-medium-articles-dataset}

**About Dataset:**

**Title:** Towards Data Science Medium Articles Dataset

**Overview:**

This dataset is a comprehensive collection of blog posts sourced from
Medium, focusing specifically on articles published under the \"Towards
Data Science\" publication. It consists of two primary columns:

-   **Title:** Contains the title of each blog post, providing a concise
    summary of the content and facilitating tasks such as title-based
    classification, keyword extraction, and trend analysis.

-   **Text:** Contains the full text content of each blog post, offering
    detailed information and insights on various topics in Data Science.
    This column serves as a valuable resource for Natural Language
    Processing (NLP) tasks including sentiment analysis, topic modeling,
    text classification, information extraction, text generation, and
    recommendation systems.

**Purpose:**

The dataset offers a comprehensive view of the current discourse within
the Towards Data Science community on Medium. It enables the exploration
of trends, popular topics, and the evolution of Data Science over time.
Additionally, it serves as a valuable resource for researchers and
practitioners in the field of NLP, providing ample opportunities for
data exploration, model training, and algorithm development.

**Use in RAG:**

The Towards Data Science Medium Articles Dataset serves as a rich source
of external knowledge for Retrieval Augmented Generation (RAG). By
leveraging the vast array of blog posts contained within the dataset,
RAG models can retrieve relevant information to augment their
understanding of input prompts or queries, leading to more accurate,
informative, and contextually relevant generated responses. This dataset
facilitates the integration of external knowledge sources into the
generation process, enhancing the capabilities of RAG models in
producing high-quality text outputs.
:::


## **Load data for RAG**
:::


``` python
# loading the cleaned dataset using CSVLoader
loader = CSVLoader(
    file_path= "/kaggle/input/1300-towards-datascience-medium-articles-dataset/medium.csv",
    csv_args={
        "delimiter": ",",
        
        "fieldnames": ["Title", "Text"],
    },
)

dataset_rag = loader.load()
```
:::


``` python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(dataset_rag )

# create the embedding function
embeddings = HuggingFaceEmbeddings(
    # This argument specifies the pre-trained model name to be used for generating embeddings.
    # Here, "sentence-transformers/all-mpnet-base-v2" is a pre-trained sentence transformer model 

    model_name="sentence-transformers/all-mpnet-base-v2",

    # set device to "cuda" to leverage the GPU for faster processing if available.
    model_kwargs={"device": "cuda"}
)
# create and store embeddings into Chroma
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="/kaggle/working/chroma_db")

```



``` python
# Create a retriever
# for threshold greater than 0.4 it was tested that not similar documents where found
retriever=vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.35, "k": 5},
        )
```

``` python
# This code creates a pipeline for text generation using a pre-trained model (model) 
# and its tokenizer (tokenizer). It leverages mixed precision (torch.bfloat16) 
# for potentially faster inference and limits generated text to 512 tokens.
pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    model_kwargs = {"torch.dtype": torch.bfloat16},
    max_new_tokens=512    
)
```
:::


## Combining Large Language Model (LLM) with Retrieval System and Text Generation

**Creating Custom LLM Pipeline (`gemma_llm`)**

-   **HuggingFacePipeline:** Wraps the existing text-generation pipeline
    (`pipeline`) into a custom `HuggingFacePipeline` class.
-   **Model Kwargs:** Sets default argument (`temperature=0.7`)
    controlling randomness of generated text.

**Building RetrievalQA Object (`qa`)**

-   **RetrievalQA.from_chain_type:** Creates a `RetrievalQA` object
    combining retrieval and question answering functionalities.
    -   `llm (TextGenerationPipeline)`: Takes `gemma_llm` for text
        generation.
    -   `chain_type (str)`: Set to `"stuff"`, defining retrieval-LM
        chain functionality.
    -   `retriever (RetrievalInterface)`: Refers to a separate retriever
        object for retrieving relevant information.

**In essence:**

This code integrates LLM text-generation capabilities with a retrieval
system, likely using retrieved information to inform text generation for
more comprehensive responses.

## Text Generation Code Breakdown

**1. Prompt Creation (`pipeline.tokenizer.apply_chat_template`)**

-   **Function:** Formats conversation history (`messages`) into a
    suitable prompt for the LLM.
-   **Arguments:**
    -   `messages (List[Dict])`: List of dictionaries representing
        conversation history.
    -   `tokenize (bool, optional)`: Set to `False`, indicating no
        tokenizer application.
    -   `add_generation_prompt (bool, optional)`: Set to `True`, adds a
        prompt specifically designed for chat-like text generation
        tasks.

**2. Text Generation with Pipeline (`pipeline`)**

-   **Function:** Continues the conversation based on the provided
    prompt.
-   **Arguments:**
    -   `prompt (str)`: Crucial prompt created in the previous step.
    -   `max_new_tokens (int, optional)`: Limits the number of tokens
        generated to 512.
    -   `add_special_tokens (bool, optional)`: Set to `True`, adds
        special tokens to the prompt.
    -   `do_sample (bool, optional)`: Set to `True`, enabling random
        sampling during generation.
    -   `temperature (float, optional)`: Controls randomness of
        generated text, set to 0.7.
    -   `top_k (int, optional)`: Restricts generation to top k most
        likely tokens, set to 10.
    -   `top_p (float, optional)`: Controls sampling process, set to
        0.95 for preferring high-probability tokens.
:::

::: {#8b48ba3e .cell .code execution_count="17" execution="{\"iopub.execute_input\":\"2024-04-14T21:49:54.719042Z\",\"iopub.status.busy\":\"2024-04-14T21:49:54.718661Z\",\"iopub.status.idle\":\"2024-04-14T21:49:54.725914Z\",\"shell.execute_reply\":\"2024-04-14T21:49:54.725214Z\"}" papermill="{\"duration\":2.2933e-2,\"end_time\":\"2024-04-14T21:49:54.727845\",\"exception\":false,\"start_time\":\"2024-04-14T21:49:54.704912\",\"status\":\"completed\"}" tags="[]"}
``` python
gemma_llm = HuggingFacePipeline(
    pipeline=pipeline,
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 512,
        "add_special_tokens": True,
        "do_sample": True,
        "top_k": 10,
        "top_p": 0.95
    },
)
# Create a RetrievalQA object
qa = RetrievalQA.from_chain_type(
    llm=gemma_llm,  # Pass the text-generation pipeline object
    chain_type="stuff",
    retriever=retriever  # retriever object
)
```
:::

::: {#4c9c47a7 .cell .code execution_count="18" execution="{\"iopub.execute_input\":\"2024-04-14T21:49:54.756128Z\",\"iopub.status.busy\":\"2024-04-14T21:49:54.755518Z\",\"iopub.status.idle\":\"2024-04-14T21:49:54.760639Z\",\"shell.execute_reply\":\"2024-04-14T21:49:54.759831Z\"}" papermill="{\"duration\":2.124e-2,\"end_time\":\"2024-04-14T21:49:54.762507\",\"exception\":false,\"start_time\":\"2024-04-14T21:49:54.741267\",\"status\":\"completed\"}" tags="[]"}
``` python
def get_answer(question):
    message = [
    {"role": "user", "content": question},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, truncation=True)
    result = qa.invoke(prompt)
    return (result['result'].split('Helpful Answer:')[1])
    
    
```
:::

::: {#91412cd6 .cell .code execution_count="19" execution="{\"iopub.execute_input\":\"2024-04-14T21:49:54.790724Z\",\"iopub.status.busy\":\"2024-04-14T21:49:54.789982Z\",\"iopub.status.idle\":\"2024-04-14T21:50:46.715635Z\",\"shell.execute_reply\":\"2024-04-14T21:50:46.714503Z\"}" papermill="{\"duration\":51.956304,\"end_time\":\"2024-04-14T21:50:46.732217\",\"exception\":false,\"start_time\":\"2024-04-14T21:49:54.775913\",\"status\":\"completed\"}" tags="[]"}
``` python
question =  "What is gradient descent?and its connection with backpropagation??"
display(Markdown(get_answer(question)))
```




### What is Gradient Descent?

- Gradient descent is an iterative optimization algorithm that finds extrebounds (or minima) of a function.
- It works by iteratively updating the parameters of the model in the direction of the negative gradient of the loss function.


### Connection with Backpropagation:

- Backpropagation is the process of calculating the gradients of a loss function with respect to the model parameters.
- It's an essential part of training supervised machine learning models, where we need to update the model parameters to minimize the loss function.


**How it works: (The process of finding gradients)**

1. **Calculate the gradients:**
    - We need to calculate the gradients of the loss function with respect to each parameter.
    - This involves traversing over the training data and measuring how each parameter contributes to the loss.


2. **Update the parameters:**
    - We update each parameter in the opposite direction of its gradient.
    - The amount of update is typically controlled by a learning rate, which determines how much we trust the gradient.


3. **Iterate until convergence:**
    - We repeat steps 1-2 until the loss function reaches a minimum.
    - This process is called "training" or "learning."


**In the context of machine learning, backpropagation is used to calculate the gradients of the loss function with respect to the model parameters. This information is then used to update the parameters and improve the model's performance.**

**Key features of backpropagation:**

- It's a recursive algorithm, meaning it repeatedly calculates the gradients of the loss function.
- It's efficient and widely used in machine learning models.
- It's essential for training supervised models.


**Applications of gradient descent and backpropagation:**

- Training supervised machine learning models (e.g., linear regression, logistic regression, classification models)
- Optimizing functions in various domains (e.g., computer vision, natural language processing)
- Solving linear systems and solving linear equations.
:::
:::


``` python
question =  "What is the significance of p-value?"
display(Markdown(get_answer(question)))
```

::: {.output .display_data}


The p-value is a statistical measure that indicates the probability of obtaining the observed data purely by chance. A p-value less than 0.05 suggests that the observed data is statistically significant and unlikely to have occurred by chance.

``` python
question="How do you build a random forest model?"
display(Markdown(get_answer(question)))
```

::: {.output .display_data}


To build a random forest model, you follow these steps:

**Step 1: Choose the base learner (tree classifier)**
- Decide on the splitting criterion for creating splits in the tree.
- Choose the maximum depth of the tree.

**Step 2: Bootstrap sampling**
- Draw random samples with replacement from the original data.
- This creates multiple training datasets.

**Step 3: Independent tree construction**
- For each bootstrap sample, build a classification or regression tree.
- At each node, randomly choose a subset of features to consider for splitting.

**Step 4: Ensemble learning**
- Combine the predictions from the individual trees to create the final prediction.
- This reduces the variance and improves the model's accuracy.

**Key parameters for building a random forest model:**

- **Number of estimators (ntree)**: The number of trees to grow.
- **Max depth** of the tree.
- **Number of features to consider** when splitting at a node.
- **Criterion for splitting** the nodes.
- **Whether to bootstrap** the samples.

**The final model is an ensemble of these trees, and their predictions are aggregated to give the final prediction. This usually improves accuracy and handles overfitting.
:::
:::


## **End of Notebook.** {#end-of-notebook}
:::


:::

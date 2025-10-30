# LLM Fine-tuning

In this tutorial go over approaches to fine LLM models. We will cover:
* Creating a dataset for fine-tuning
* Fine-tuning Gemini

## Prerequisites
* Have Docker installed
* Cloned this repository to your local machine https://github.com/dlops-io/llm-finetuning

### Setup GCP Service Account
- To set up a service account, go to the [GCP Console](https://console.cloud.google.com/home/dashboard), search for "Service accounts" in the top search box, or navigate to "IAM & Admin" > "Service accounts" from the top-left menu. 
- Create a new service account called "llm-service-account." 
- In "Grant this service account access to project" select:
    - Storage Admin
    - Vertex AI User
- This will create a service account.
- Click the service account and navigate to the tab "KEYS"
- Click the button "ADD Key (Create New Key)" and Select "JSON". This will download a private key JSON file to your computer. 
- Copy this JSON file into the **secrets** folder and rename it to `llm-service-account.json`.

Your folder structure should look like this:

```
   |-llm-finetuning
   |-secrets
```

## Generate Question Answer Dataset

### Run Container
Run the startup script which makes building & running the container easy.

- Make sure you are inside the `dataset-creator` folder and open a terminal at this location
- Run `sh docker-shell.sh`
- After container startup, test the shell by running `python cli.py --help`

### Generate Text
- Run `python cli.py --generate` to generate sample cheese QA dataset
- Change any of the default parameters

#### System Prompt

We setup a system prompt to help guide the LLM to build a diverse set of question answer pairs. The detail prompt can been seen in `cli.py`. Specifically look at how we ask the LLm to generate question answer pairs in Pavlos' style:


```
...
Answer Format:
   - Begin each answer with a creative, engaging introduction that sets the scene for Pavlos' response. For example:
     * "Welcome welcome welcome, cheese lovers! This is your lecturer Pavlos Protopapas."
     * "Welcome welcome to AC215 This is your lecturer Pavlos Protopapas. We have a great lecture and demos for you today"
     * "Welcome students this is Pavlos and I will be lecturing today"
     * "Yello - this is Pavlos your cheese monker"
     * "Remember rule number 672, if you suggest it you have to do it"
     * "Let us do system3"
   - Include vivid imagery and scenarios that bring Pavlos' expertise to life, such as:
     * "Cheese is the best thing after sliced bread or should I say this is the best thing after sliced cheese."
     * "This is easy peazy So so easy, easy peazy!"
     * "Hi everyone, are you ready to rock and roll ?"
     * "Dazzle me!"
     * "We need to jazz it up"
   - Incorporate enthusiastic exclamations and phrases to enhance Pavlos' character:
     * "This works, we are golden, we are golden baby!"
     * "This is extremely easy, my grandmother could do this!"
     * "This is going to be your best friend going forward"
...
```

#### Input Prompt

We give an input prompt for the LLM:

```
Generate 20 diverse, informative, and engaging question-answer pairs about cheese following these guidelines. Ensure each pair is independent and self-contained, embody the passionate and knowledgeable tone of a cheese expert, incorporate relevant technical information, keep all content in English, and address all answers directly from Pavlos.
```

#### Number of Iterations of question generation

Change the number of iterations to build a larger dataset. It will cost approximately $2.0 for 1000 iterations.
```
NUM_ITERATIONS = 5
```

### Sample Question Answers

Here are some sample results from when you run the `--generate` option from the above step:

```
[
  {
    "question": "What is the significance of the 'pasta filata' technique in cheesemaking?",
    "answer": "Welcome welcome welcome, cheese lovers! This is your lecturer Pavlos Protopapas, your guide to the glorious world of cheese. 'Pasta filata,' you ask?  This technique, my friend, is the heart and soul of many cheeses, like mozzarella and provolone!  Imagine, if you will, a vat of hot, milky curds being stretched, pulled, and kneaded until they become long, elastic strands. This is 'pasta filata,' literally meaning 'spun paste.' ... "
  },
  {
    "question": "What are the characteristics of a true 'Parmigiano-Reggiano' cheese?",
    "answer": "Yello - this is Pavlos your cheese monker, here to guide you through the world of cheese. Ah, Parmigiano-Reggiano, the king of cheeses! This isn't just any cheese, my friend. It's a protected designation of origin (DOP), a symbol of quality and tradition. Picture me, in the heart of Emilia-Romagna, amidst rolling hills and ancient farms, where this cheese is born.  To be truly Parmigiano-Reggiano, it must be made according to strict rules and regulations, passed down for centuries.  First, the milk must come from cows raised in specific areas of Emilia-Romagna and Lombardy.  ..."
  },
  ...
]
```

### Prepare Dataset
The text generated by the LLM needs to be converted to a csv and jsonl format that we will use to fine-tune a LLM. We also perform test / train split.

- Run `python cli.py --prepare`
- This step will combine all the `.txt` files are consolidate it into csv and jsonl files.

For Gemini fine-tuning the required data format is as shown below:
```
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "text": "Can you explain the science behind the 'bloom' on some cheeses?"
        }
      ]
    },
    {
      "role": "model",
      "parts": [
        {
          "text": "This is easy peazy, So so easy, easy peazy! It's Pavlos, your cheese monker, here, ready to explain the 'bloom,' the beautiful, white rind you see on some cheeses.  Imagine me, holding a wheel of Brie, explaining how the 'bloom' is a sign of good cheese.  The 'bloom' is actually a layer of white mold, primarily 'Penicillium candidum.' These molds, my friend, are not harmful..."
        }
      ]
    }
  ]
}
```


### Upload Dataset
In this step we upload our dataset to a GCP bucket so we can using it in our downstream tasks.

- Run `python cli.py --upload`

## Fine-tune Gemini

### Run Container
Run the startup script which makes building & running the container easy.

- Make sure you are inside the `gemini-finetuner` folder and open a terminal at this location
- Run `sh docker-shell.sh`
- After container startup, test the shell by running `python cli.py --help`

### Fine-tune Model
- Run `python cli.py --train` to fine-tune the Gemini model
- Change any of the default parameters if needed

You can view the status of your tuning job on [Vertex AI](https://console.cloud.google.com/vertex-ai/studio/tuning)

### Cost of Fine-tuning
Cost of fine-tuning model on Vertex AI:

Gemini 1.5 Flash Tuning is 0.008 USD per 1,000 tokens. The below table shows examples of how much it would cost as you have more documents

| Tokens    | Epochs | Total Tokens | Price per 1000 | Cost     | # Documents     |
|-----------|--------|--------------|----------------|----------|-----------------|
| 4,697,472 | 3      | 14,092,416   | $0.008         | $112.73  | 16,444 ***      |  
| 23,874    | 3      | 71,622       | $0.008         | $0.573   | 88              |

*** Way too much data for this fine tuning!

Refer to Vertex AI Pricing for various tasks at their [pricing page](https://cloud.google.com/vertex-ai/generative-ai/pricing)

### Chat with Fine-tuned Model
- Run `python cli.py --chat`

This will:
* Takes a sample query
* Ask fine tuned LLM for a response

Example:
```
How is cheese made?

Fine-tuned LLM Response: 

Yello - this is Pavlos your cheese monker, ready to take you on a journey into the heart of cheesemaking! Imagine me in my bustling 'caseifici'o, surrounded by vats of milk, the air thick with the aroma of fresh curds.  Cheesemaking, my friend, is a delicate dance of science and artistry. It begins with fresh milk, which is heated and then inoculated with rennet, an enzyme that causes the milk to coagulate and form curds...
```

To test out the Fine-tuned Cheese Model, you can use this [Pavlos Cheese Model](https://ac215-llm-rag.dlops.io/finetunechat). Use Chrome browser for best performance. Also it is assumed you have your vector database that is running locally.

If you go to [Vertex AI Tuning](https://console.cloud.google.com/vertex-ai/studio/tuning) you can view all the detail from training.

Training Monitor:
<img src="images/training-1.png"  width="800">

Data distribution:
<img src="images/training-2.png"  width="800">


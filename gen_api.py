# text ="A large language model (LLM) is a type of computational model designed for natural language processing tasks such as language generation. As language models, LLMs acquire these abilities by learning statistical relationships from vast amounts of text during a self-supervised and semi-supervised training process.The largest and most capable LLMs are artificial neural networks built with a decoder-only transformer-based architecture, enabling efficient processing and generation of large-scale text data. Modern models can be fine-tuned for specific tasks, or be guided by prompt engineering. These models acquire predictive power regarding syntax, semantics, and ontologies inherent in human language corpora, but they also inherit inaccuracies and biases present in the data on which they are trained"



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import uvicorn


# Initialize FastAPI
app = FastAPI()

# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    txt: str


@app.post("/test")
async def summarize_text(request: TextRequest):
    # Define the prompt for summarization
    prompt = f"Could you list down the techstack: {request.txt}"
    genai.configure(api_key="xxxxxxxxxxxxxxxx")

    try:
        # Initialize the Generative AI model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate the summary
        response = model.generate_content(prompt)
        
        final_response = {"summary":response.text}
        print(final_response)
        # Return the response in JSON format
        return final_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

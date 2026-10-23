def ov_question(prompt):
    return_q = f"""This image was generated from the prompt: {prompt}. 
    Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image. 

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

    When the answer is false, you must:
    1. Identify the main error and describe it briefly in "explanation".
    2. In "edit_prompt", provide a **concrete image editing instruction** to fix the error.  
    - The instruction must specify the exact action (e.g., add / remove / replace / move).  
    - The instruction must specify the location or reference point (e.g., "delete the bottle in the bottom-right corner", "add a dog next to the left pillar").  
    - Do not give vague instructions such as "add more bottles" or "ensure the count is correct". Be precise and actionable.  

    Respond strictly in the following JSON format: """ + """

    {
        "answer": true/false,
        "explanation": "If the answer is false, briefly summarize the main error.",
        "edit_prompt": "If the answer is false, provide a concrete and location-specific editing instruction."
    }
    """
    
    return return_q

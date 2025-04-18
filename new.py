# import time
# import requests

# batch_id = "35af90f7-042d-49fb-8b3e-92aa2c49e336"
# url = f"https://speech-analysis-api-531866504854.us-central1.run.app/analyze-batch"

# while True:
#     print(f"Checking batch status for ID: {batch_id}...")
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         data = response.json()
#         print("Batch Complete!")
#         print(data)
#         break
#     else:
#         print("Still processing... waiting 10 seconds")
#         time.sleep(10)

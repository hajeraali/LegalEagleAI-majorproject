# LegalEagleAI-majorproject
befor you run project please open postgresql pg admin and in query tab execute this query
**ALTER TABLE public.clientappointments
ADD COLUMN barcouncil_id VARCHAR(100);**


run below code to set up project
1. pip install flask flask-session flask-mailman psycopg2 pandas nltk transformers werkzeug PyPDF2 docx google-generativeai python-dotenv
2. python -c "import nltk; nltk.download('all');"
3. keep running python sync_bookings_to_firebase.py in background
4. **to see client and lawyer appointments simultaneously open the project in 2 diff web windows**
5. **IMP NOTE PLEASE USE SAME NAME AS LOGGED IN CLIENT AND EMAIL FOR APPOINTMENTS**

admin page info:
username: admin
password: admin123

for each client use password : *Cli123
for each lawyer use password: *Law1234

new file created to sync entries from postgresql needed to be run in background on different terminal using command : "python sync_bookings_to_firebase.py"

.env and 3 more files is updated on mail and project drive download and add them to project folder

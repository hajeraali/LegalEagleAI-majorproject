document.getElementById('appointmentForm').addEventListener('submit', async (e) => {
    e.preventDefault(); // Prevents page from refreshing

    // Retrieve form data
    const appointmentDate = document.getElementById('appointmentDate').value;
    const clientName = document.getElementById('clientName').value;
    const clientEmail = document.getElementById('clientEmail').value;
    const appointmentTime = document.getElementById('appointmentTime').value;
    const caseDetails = document.getElementById('caseDetails').value;

    const appointmentData = {
        appointmentDate,
        clientName,
        clientEmail,
        appointmentTime,
        caseDetails
    };

    try {
        const response = await fetch('/book-appointment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(appointmentData)
        });

        const result = await response.json();

        if (response.ok) {
            alert(result.message);
            document.getElementById('appointmentForm').reset(); // Clear form
        } else {
            alert('Error booking appointment: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('There was an error booking the appointment.');
    }
});

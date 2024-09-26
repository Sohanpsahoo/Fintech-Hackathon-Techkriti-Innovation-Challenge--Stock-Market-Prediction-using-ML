function predictPrice() {
    // Get the input date value
    var inputDate = document.getElementById('date').value;

    // You can perform any client-side validation or preprocessing here

    // Send an asynchronous request to your Flask server to handle the form submission
    fetch('/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'date=' + encodeURIComponent(inputDate),
    })
    .then(response => response.text())
    .then(data => {
        // Update the HTML with the response from the server
        document.body.innerHTML = data;
    })
    .catch(error => console.error('Error:', error));
}

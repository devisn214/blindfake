document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // Display the "Processing..." message
    document.getElementById('result').innerHTML = "<p>Processing... Please wait.</p>";
    document.getElementById('downloadLinks').style.display = 'none';

    // Prepare the form data
    const formData = new FormData();
    formData.append('file', document.getElementById('fileInput').files[0]);

    // Send the form data to the backend
    fetch('/detect', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Once the prediction is done, display the download links
        document.getElementById('downloadLinks').style.display = 'block';

        // Set the download links
        document.getElementById('textReportLink').href = data.text_report_path;
        document.getElementById('brailleReportLink').href = data.braille_report_path;
        document.getElementById('audioReportLink').href = data.audio_report_path;

        document.getElementById('result').innerHTML = "<p>Reports are ready to download.</p>";
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = "<p>Error processing the file. Please try again.</p>";
    });
});

// Clear previous result when a new file is selected
document.getElementById('fileInput').addEventListener('change', function() {
    document.getElementById('result').innerHTML = "";
    document.getElementById('downloadLinks').style.display = 'none';
});


window.onload = function() {
 trackTransferStatus()
};

function getTransferStatus() {

    console.log("receiver here");
    if (window.XMLHttpRequest) {
        xmlhttp = new XMLHttpRequest();
    } else {
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }

    xmlhttp.onreadystatechange = function () {
        if (this.readyState === 4 && this.status === 200) {

            if (this.responseText.localeCompare("1") === 0) {
                window.location.replace("result")
            }
            document.getElementById("transfer_status").innerHTML =
                this.responseText;
        }
    };

    xmlhttp.open("GET", "get_status", true);
    xmlhttp.send();

}

function trackTransferStatus() {
    console.log("caller online");
    window.setInterval(getTransferStatus, 5000)
}
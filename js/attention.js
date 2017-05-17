(function() {
    "use strict";

    function request(url, callback) {
        var xhr = new XMLHttpRequest();
        xhr.onload = e => callback(JSON.parse(xhr.response));
        xhr.open("GET", url);
        xhr.send();
    }

    var colors = {
        ANGER:   [255, 32,  88],  // Red
        SADNESS: [59,  163, 252], // Blue
        JOY:     [171, 216, 0]    // Light green
    }

    function rgba(rgb, alpha) {
        return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
    }

    var key = document.getElementById("attention-key");
    let i = 0;
    for (let colorName in colors) {
        let label = document.createElement("span")
        label.textContent = colorName;
        label.style.backgroundColor = rgba(colors[colorName], 1.0);
        if (i > 0) {
            key.appendChild(document.createTextNode(" / "));
        }
        key.appendChild(label);
        i++;
    }

    var attentions = document.querySelectorAll(".attention");
    for (let parent of attentions) {
        request(parent.dataset.filepath, obj => {
            let bq = document.createElement("blockquote");
            for (let k = 0; k < obj.words.length; k++) {
                let span = document.createElement("span");
                span.textContent = obj.words[k];
                span.style.backgroundColor = rgba(colors[obj.pred_label], obj.attention[k]);
                span.className = "token";
                bq.appendChild(span);
                bq.appendChild(document.createTextNode(" "));
            }
            parent.appendChild(bq);
        });
    }
}());
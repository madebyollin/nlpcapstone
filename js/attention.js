(function() {
    function request(url, callback) {
        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function() {
            if (xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200) {
                callback(JSON.parse(xhr.responseText));
            }
        }
        xhr.open("GET", url, true);
        xhr.send(null);
    }

    var colors = {
        ANGER:   [255, 32,  88],  // Red
        SADNESS: [59,  163, 252], // Blue
        JOY:     [171, 216, 0]    // Light green
    }

    function rgba(rgb, alpha) {
        return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
    }

    var nodes = document.querySelectorAll(".attention");
    for (let i = 0; i < nodes.length; i++) {
        let node = nodes[i];
        request(node.dataset.filepath, obj => {
            let max = Math.max(...obj.attention);
            let min = Math.min(...obj.attention);
            let rng = max - min;
            for (let k = 0; k < obj.words.length; k++) {
                let span = document.createElement("span");
                span.textContent = obj.words[k];
                span.style.backgroundColor = rgba(colors[obj.pred_label], (obj.attention[k] - min) / (rng));
                span.style.padding = "2px";
                node.appendChild(span);
                node.appendChild(document.createTextNode(" "));
            }
        });
    }
}());
$(function () {
    console.log("here");
    var dropdown_loss = $('#loss_function');
    var dropdown_optim = $('#optimizer');

    // ID of the loss function selected
    var selectedLossFunction;
    $(".loss a").click(function(){
        selectedLossFunction = $(this)[0].id;
        dropdown_loss.text($(this).text());
        dropdown_loss.val($(this).text());
    });

    var selectedOptimizer;
    $(".optim a").click(function(){
        if (typeof selectedOptimizer !== 'undefined') {
            $('#hyperparameters-form').find("tr."+selectedOptimizer).addClass("hidden");
        }
        selectedOptimizer = $(this)[0].id;
        dropdown_optim.text($(this).text());
        dropdown_optim.val($(this).text());
        $('#hyperparameters-form').find("tr."+selectedOptimizer).removeClass("hidden");

    });

    $('#hyperparameters-form').on('submit', function (e) {
        console.log(selectedLossFunction);
        var formData = {};
        try {
            //empty the hidden parameters
            $('#hyperparameters-form').find("tr.hidden input").each(function (index, node) {
                node.value = "";
            });
            //and their comments
            $('#hyperparameters-form').find("tr.hidden textarea").each(function (index, node) {
                node.value = "";
            });


            $('#hyperparameters-form').find("input").each(function (index, node) {
                formData[node.id] = {};
                formData[node.id]["value"] = node.value;
            });
            formData["loss_function"] = {};
            formData["loss_function"]["value"] = selectedLossFunction;

            formData["optimizer"] = {};
            formData["optimizer"]["value"] = selectedOptimizer;


            $('#hyperparameters-form').find("tr textarea").each(function (index, node) {
                //formData[node.id]["comments"] ="" ;
                console.log(node);

                var value = formData[node.id]["value"];
                var comment = node.value;

                formData[node.id] = {}
                formData[node.id]["comments"] = comment;
                formData[node.id]["value"] = value;

                console.log(formData);
                // formData[node.id]["comments"] = node.value;
            });
        } catch (error) {
            console.error(error);
        }

        /*const data = new URLSearchParams();
        console.log($(this));
        for (const pair of new FormData($(this))) {
            console.log(pair)
            data.append(pair[0], pair[1]);
        }*/
        console.log(formData);
        fetch("http://0.0.0.0:5001/run_model",
        {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify(formData),
            mode: 'no-cors'
        })
        .then(function(res){

            // $("#main-content").fadeOut(2000);
            // $("#viz_link_container").fadeIn(2000);

        })
        .then(function(data){ console.log( "aas"+JSON.stringify( data ) ) });

    })

    $("#hyperparameters-form").submit(function(e) {
        e.preventDefault();
    });
});

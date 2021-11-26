// 'use strict';

import React from "react";
import ReactDOM from "react-dom";
<<<<<<< HEAD
<<<<<<< HEAD
import RevealsPage from "../jsx/reveals";

let url = window.location.href.toLowerCase();
console.log(url);
if (url.includes("/reveals")) {
    ReactDOM.render(<RevealsPage />, document.getElementById("root"));
}
=======
import App from "../jsx/App";

$(function () {

    var form = $('#address-form');
    form.submit(e => {
        e.preventDefault();
        console.log('awefwef');
        const a = $( "#input-address" ).first().val()
        window.location.href = "/cluster?address=" + a;
    })
});

ReactDOM.render(<App />, document.getElementById('root'));
>>>>>>> b4ef37e (add react (unfinished))
=======
import RevealsPage from "../jsx/reveals";

let url = window.location.href.toLowerCase();
console.log(url);
if (url.includes("/reveals")) {
    ReactDOM.render(<RevealsPage />, document.getElementById("root"));
}
>>>>>>> cd12d85 (half react)

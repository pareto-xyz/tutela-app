// 'use strict';

import React from "react";
import ReactDOM from "react-dom";
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

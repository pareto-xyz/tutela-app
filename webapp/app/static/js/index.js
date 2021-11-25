// 'use strict';

<<<<<<< HEAD
//this file is bundled. 
=======
import React from "react";
import ReactDOM from "react-dom";
import App from "../jsx/App";
>>>>>>> b4ef37e (add react (unfinished))

import React from "react";
import ReactDOM from "react-dom";

import TransactionPage from "../jsx/transaction";
import ClusterPage from "../jsx/address";


let url = window.location.href.toLowerCase();
if (url.includes('/cluster')) {
    ReactDOM.render(<ClusterPage />, document.getElementById('address-page'));
}
else if (url.includes("/transaction")) {
    ReactDOM.render(<TransactionPage />, document.getElementById("root"));
}

<<<<<<< HEAD
=======
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

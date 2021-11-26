// 'use strict';

//this file is bundled. 

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


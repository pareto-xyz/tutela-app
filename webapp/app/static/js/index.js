// 'use strict';

//this file is bundled. 

import React from "react";
import ReactDOM from "react-dom";

import TransactionPage from "../jsx/transaction";
import ClusterPage from "../jsx/address";


let url = new URL(window.location.href.toLowerCase());
const params = url.searchParams;
if (url.pathname === '/cluster') {
    ReactDOM.render(<ClusterPage params={params} />, document.getElementById('address-page'));
}
else if (url.pathname === "/transaction") {
    ReactDOM.render(<TransactionPage params={params} />, document.getElementById("root"));
}


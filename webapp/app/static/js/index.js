// 'use strict';

//this file is bundled. 

import React from "react";
import ReactDOM from "react-dom";

import TransactionPage from "../jsx/transaction";
import ClusterPage from "../jsx/address";
import HomePage from "../jsx/home";


let url = new URL(window.location.href.toLowerCase());
const params = url.searchParams;
const root = document.getElementById('root');
if (url.pathname === '/' || url.pathname === '/index') {
    ReactDOM.render(<HomePage params={params} />, root);
}
else if (url.pathname === '/cluster') {
    ReactDOM.render(<ClusterPage params={params} />, root);
}
else if (url.pathname === "/transaction") {
    ReactDOM.render(<TransactionPage params={params} />, root);
}


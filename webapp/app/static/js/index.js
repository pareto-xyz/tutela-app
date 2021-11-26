// 'use strict';

import React from "react";
import ReactDOM from "react-dom";
import RevealsPage from "../jsx/reveals";

let url = window.location.href.toLowerCase();
console.log(url);
if (url.includes("/reveals")) {
    ReactDOM.render(<RevealsPage />, document.getElementById("root"));
}

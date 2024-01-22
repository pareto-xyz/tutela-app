import React from 'react';
import { useEffect } from "react";

function getDisplayText(link, current) {
    if (current === link) {
        return (
            <strong>{link}</strong>
        );
    } else {
        return link;
    }
}

export default function Header(props) {
    const { current } = props;
    useEffect(() => {
        window.addEventListener("load", function () {
            let hamb = document.querySelector(".navbar-toggler");
            hamb.addEventListener("click", function () {
                if ($(".nav.scrolled")) {
                    $(".nav").toggleClass("scrolled", !document.querySelector(".navbar-collapse").classList.contains("show"));
                }
            });
        });
        window.addEventListener("scroll", function () {
            if ($(".navbar-collapse.show")) {
                $(".nav").toggleClass("scrolled", $(this).scrollTop() > 40);
            }
        });
        return () => {
            window.removeEventListener("load", function () {
                let hamb = document.querySelector(".navbar-toggler");
                hamb.removeEventListener("click", function () {
                    if ($(".nav.scrolled")) {
                        $(".nav").toggleClass("scrolled", !document.querySelector(".navbar-collapse").classList.contains("show"));
                    }
                });
            });
            window.removeEventListener("scroll", function () {
                if ($(".navbar-collapse.show")) {
                    $(".nav").toggleClass("scrolled", $(this).scrollTop() > 40);
                }
            });
        }
    }, []);
    return (
        <header className="nav fixed col-12 navbar navbar-expand-lg">
            <a className="header-logo" href="https://tornado.cash/"><img width="50" src="/static/img/tornado_logo.svg"
                alt="logo"></img></a>
            <button
                className="navbar-toggler"
                type="button"
                data-toggle="collapse"
                data-target="#navbarSupportedContent"
                aria-controls="navbarSupportedContent"
                aria-expanded="false"
                aria-label="Toggle navigation"
            >
                <span className="navbar-toggler-icon">
                    <i className="fas fa-bars fa-1x"></i>
                </span>
            </button>
            <div className="my-2 my-lg-0 collapse navbar-collapse col-md-12 col-lg-6" id="navbarSupportedContent">
                <ul className="navbar-nav">
                    <li className="nav-item">
                        <a className="nav-link" href="/#">{getDisplayText('home', current)}</a>
                    </li>
                    <li className="nav-item">
                        <a className="nav-link" href="/#about">{getDisplayText('about', current)}</a>
                    </li>
                    <li className="nav-item">
                        <a className="nav-link" href="/#reveals">{getDisplayText('reveals', current)}</a>
                    </li>
                    <li>
                        <a className="nav-link"
                            href="https://etherscan.io/address/0xFca529E4Fd732C6a94FDEbE3Ad32FC2975e759b3/">donations <i className="fas fa-heart"></i></a>
                    </li>
                </ul>
            </div>
        </header>
    )
}
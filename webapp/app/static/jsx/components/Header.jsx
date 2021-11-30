import React from 'react';

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
    const {current} = props;
    return (
        <header className="container">
            <a className="header-logo" href="https://tornado.cash/"><img width="50" src="/static/img/tornado_logo.svg"
                alt="logo"></img></a>
            <div className="nav">
                <a className="nav-link" href="/">{getDisplayText('home', current)}</a>
                <a className="nav-link" href="/cluster">{getDisplayText('address', current)}</a>
                <a className="nav-link" href="/#about">{getDisplayText('about', current)}</a>
                <a className="nav-link" href="/#reveals">{getDisplayText('reveals', current)}</a>
            </div>
        </header>
    )
}
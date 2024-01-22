import React from 'react';
import { Dropdown } from 'react-bootstrap';

export default function ChooseTornadoPool() {
    return (
        <Dropdown className="col-sm-8 col-md-6 col-lg-4 investigate">
            <Dropdown.Toggle variant="dark" size="md">
                <img width="20px" src="static/img/tornado_logo.svg"></img> investigate a tornado cash pool
            </Dropdown.Toggle>
        </Dropdown>
    )
}
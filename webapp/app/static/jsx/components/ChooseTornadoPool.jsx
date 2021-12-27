import React, { useEffect, useState } from 'react';
import { Dropdown } from 'react-bootstrap';
import { getApi } from './utils';

export default function ChooseTornadoPool() {
    const [poolOptions, setPoolOptions] = useState([]);

    const onSelect = (addr) => {
        window.location.href = "/cluster?address=" + addr;
    }

    useEffect(() => {
        getApi('/utils/gettornadopools', response => {
            const pools = response.data.data.pools;
            setPoolOptions(pools);
        })
    }, [])


    return (
        <Dropdown className="col-sm-12 col-md-6 col-lg-4 button-group" onSelect={onSelect}>
            <Dropdown.Toggle variant="dark" size="md" >
            <img width="20px" src="static/img/tornado_logo.svg"></img> investigate a tornado cash pool
            </Dropdown.Toggle>

            <Dropdown.Menu className="col-sm-12 col-md-6 col-lg-4 "  >
                {poolOptions  && poolOptions.map(({address, name}) => {
                    return (
                        <Dropdown.Item eventKey={address} key={address} >
                            {name} pool
                        </Dropdown.Item>
                    )
                })}
            </Dropdown.Menu>
        </Dropdown>
    )
}
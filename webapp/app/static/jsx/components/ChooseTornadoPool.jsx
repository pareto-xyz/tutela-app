import React, { useEffect, useState } from 'react';
import { Dropdown } from 'react-bootstrap';
import { getApi } from '../../js/utils';

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
        <Dropdown className="col-sm-8 col-md-6 col-lg-4 investigate" onSelect={onSelect}>
            <Dropdown.Toggle variant="dark" size="md">
                <img width="20px" src="static/img/tornado_logo.svg"></img> investigate a tornado cash pool
            </Dropdown.Toggle>
            <div className="col-12 address-drop">
                <div className="row">
                    <Dropdown.Menu className="col-12">
                        <div className="row">
                            <div className="col-12">
                                {poolOptions && poolOptions.map(({ address, name }) => {
                                    return (
                                        <Dropdown.Item className="pool-drop" eventKey={address} key={address} >
                                            {name} pool
                                        </Dropdown.Item>
                                    )
                                })}
                            </div>
                        </div>
                    </Dropdown.Menu>
                </div>
            </div>
        </Dropdown>
    )
}
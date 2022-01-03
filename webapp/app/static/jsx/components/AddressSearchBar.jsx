import axios from 'axios';
import React, { useEffect, useState, useRef } from 'react';
import { InputGroup, FormControl, Form, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { isValid } from './utils';
//import isTornado from '../../data/istornado';

export default function AddressSearchBar({ onSubmit, inputAddress, setInputAddress }) {

    const inputEl = useRef(null);
    const [invalid, setInvalid] = useState(false);
    const [tornadoTooltip, setTornadoTooltip] = useState('');

    useEffect(() => {
        if (!setInputAddress || inputAddress === undefined) {
            [inputAddress, setInputAddress] = useState('');
        }
    }, [])

    useEffect(() => {
        if (inputAddress.length > 0) {
            inputEl.current.value = inputAddress;
        }
    }, [inputAddress])

    const submitInputAddress = addr => {

        if (!isValid(addr)) {
            setInvalid(true);
            return;
        }
        setInvalid(false);
        onSubmit(addr);
    }

    const onChangeInputAddress = e => {
        const addr = e.target.value;
        e.preventDefault();
        setInputAddress(addr);
        axios.get('/utils/istornado?address=' + addr).then(response => {
            // response = isTornado;
            const { data, success } = response.data;
            if (success === 0) return;
            const { is_tornado, amount, currency } = data;
            if (is_tornado) {
                setTornadoTooltip(`This corresponds to the ${amount} ${currency} Tornado Cash pool.`)
                const timer = setTimeout(() => {
                    setTornadoTooltip('');
                }, 4000);
            } else {
                if (tornadoTooltip.length > 0) {
                    //was previously a tornado tooltip
                    setTornadoTooltip('');
                }
            }
        });
    }

    const renderTcashTooltip = (props) => (
        <Tooltip {...props}>
            <img width="20px" src="static/img/tornado_logo.svg"></img>{tornadoTooltip}
        </Tooltip>
    )

    return (
        <>
            <OverlayTrigger
                show={tornadoTooltip.length > 0}
                placement="right-start"
                overlay={renderTcashTooltip}
            >

                    <InputGroup onSubmit={submitInputAddress} className="fixed-width mb-1 col-sm-12 col-md-8" hasValidation>
                        <FormControl onKeyPress={(e) => {
                            if (e.key !== 'Enter') return;
                            e.preventDefault();
                            submitInputAddress(inputEl.current.value);
                        }}
                            onChange={onChangeInputAddress}
                            placeholder='eg. 0x000000000000000.........., {ens_name}.eth, etc. '
                            className="search-bar"
                            isInvalid={invalid}
                            ref={inputEl}
                        >
                        </FormControl>

                        <InputGroup.Text onClick={submitInputAddress} className="right-submit-icon"><img width="15" src="/static/img/loupe.svg" alt="search"></img> </InputGroup.Text>
                        <Form.Control.Feedback type="invalid">
                            Please enter a valid ethereum address or .eth ens name.
                        </Form.Control.Feedback>
                    </InputGroup>
            </OverlayTrigger>

        </>


    )
}
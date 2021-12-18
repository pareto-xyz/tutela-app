import axios from 'axios';
import React, { useEffect, useState, useRef } from 'react';
import { InputGroup, FormControl, Form, Overlay, Tooltip } from 'react-bootstrap';
import { isValid } from './utils';

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
        if (inputEl.current.value.length === 0 && inputAddress.length > 0) {
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
            const { data, success } = response;
            if (success === 0) return;
            const { is_tornado, amount, currency } = data;
            if (is_tornado ) {
                setTornadoTooltip(`This corresponds to the ${amount} ${currency} Tornado Cash pool.`)
            } else {
                if (tornadoTooltip.length > 0) {
                    //was previously a tornado tooltip
                    setTornadoTooltip('');
                }
            }
        });
    }

    return (
        <>
            <Overlay target={inputEl.current} show={tornadoTooltip} placement="right-start">
                {(props) => (
                    <Tooltip  {...props}>
                        {tornadoTooltip}
                    </Tooltip>
                )}
            </Overlay>
            <InputGroup onSubmit={submitInputAddress} className="mb-3 fixed-width" hasValidation>
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
        </>


    )
}
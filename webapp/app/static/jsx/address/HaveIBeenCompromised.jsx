import axios from 'axios';
import React, { useState, useRef } from 'react';
import { InputGroup, FormControl, Form, } from 'react-bootstrap';
import { compromisedTcashAddrResponse, failedCompromisedTcashResponse } from '../../data/ihavebeencompromised';
import ListOfResults from '../components/ListOfResults';


export default function HaveIBeenCompromised({ tcashAddr, aliases }) {
    const inputEl = useRef(null);
    const [val, setVal] = useState('');
    const [numCompromised, setNumCompromised] = useState(null);
    const [compromisedTxns, setCompromisedTxns] = useState(null);
    const [invalid, setInvalid] = useState(false);

    const checkIfCompromised = address => {
        axios.get('/search/compromised?address=' + address + '&pool=' + tcashAddr).then(response => {
            //  response = compromisedTcashAddrResponse;
            const { success, data } = response.data;
            if (!success) {
                setNumCompromised(null);
                setCompromisedTxns(null);
                setInvalid(true);
                return;
            }
            const { compromised_size, compromised } = data;
            setNumCompromised(compromised_size);
            setCompromisedTxns(compromised);
            setInvalid(false);
        })
    }

    return (
        <div className="query-info ">
            <div className="panel-sub">
                check if your transactions have been compromised
            </div>
            <div className="panel-title">
                YOUR COMPROMISED TXNS IN THIS POOL
            </div>
            <InputGroup  >
                <FormControl className="rounded specific-result"
                    placeholder="enter deposit address to check for compromised txs enter a deposit or withdrawal address used with this pool"
                    onChange={(e) => {
                        e.preventDefault();
                        setVal(e.target.value);
                    }}
                    onKeyPress={(e) => {
                        if (e.key !== 'Enter') {
                            return;
                        }
                        checkIfCompromised(val);
                    }}
                    ref={inputEl}
                    isInvalid={invalid}
                />
            <Form.Control.Feedback type='invalid'>Please input a valid deposit address.</Form.Control.Feedback>

            </InputGroup>


            {compromisedTxns && <ListOfResults
                sectionHeader={numCompromised !== null && <div className="results-section col-12">Total deposits compromised: {numCompromised}</div>}
                rowTitle='transaction'
                rowBadge='heuristic'
                results={compromisedTxns}
                aliases={aliases}
            />}
        </div>

    )
}
import React, { useState } from 'react';
import SortAndFilters from '../components/SortAndFilters';
import Pagination from '../components/Pagination';
import { Accordion } from 'react-bootstrap';
import AgnosticTable from '../components/AgnosticTable';

function NoClusters() {
    return (
        <div>
            <div className="center-inside">
                No clusters found.
                <br />
                <img width="200" src="/static/img/spy.png" />
                <br />
                <div>you're anonymous, harry!</div>
            </div>
        </div>
    )
}


export default function ClusterResults(props) {
    const { results, loading, refineData, getNewResults, aliases } = props;
    const noResults = results.length == 0;

    function Row({ result, idx }) {
        const { address } = result;
        const [selected, setSelected] = useState(false);
        return (
            <Accordion.Item eventKey={idx} className={selected && 'selected-result'} key={idx}>
                <Accordion.Header className="accordion-header" onClick={() => setSelected(!selected)}>
                    <div>{address}</div>
                    <div className="squashed-row">
                        <div className="accordion-badge">{result.entity}</div>
                        <div className="expand-symbol">{selected ? '-' : '+'}</div>
                    </div>
                </Accordion.Header>
                <Accordion.Body className="accordion-body">
                    <div className="panel-sub">linked address #{idx + 1}</div>
                    <AgnosticTable keyValues={Object.entries(result)} toIgnore={new Set(['address', 'id'])} aliases={aliases} />
                </Accordion.Body>
            </Accordion.Item>
        );
    }

    return (
        <div >
            <div className="results">
                <div className="panel-title">
                    LINKED ADDRESSES
                </div>
            </div>
            <SortAndFilters refineData={refineData} getNewResults={getNewResults} />
            {!noResults && <Pagination />}

            <div >
                {(loading || noResults) && <div className="results">

                    {loading ? <div id="spinner" className="justify-content-center">
                        <div className="spinner-border" role="status">
                            <span className="sr-only">Loading...</span>
                        </div>
                    </div> : (noResults && <NoClusters />)
                    }
                </div>}
                <Accordion className="overall-accordion" >
                    {results.map((result, idx) => <Row result={result} idx={idx}></Row>)}
                </Accordion>
            </div>
        </div>
    )
}
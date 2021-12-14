import React, { useState } from 'react';
import { Accordion } from 'react-bootstrap';
import AgnosticTable from './AgnosticTable';


export default function AccordionOfResults(props) {
    const { results, loading, aliases, rowTitle, rowBadge, sectionHeader,
            noDataComponent, SortAndFilters, Pagination } = props;
    const noResults = results.length == 0;

    function Row({ result, idx }) {
        const title = result[rowTitle];
        let badge = result[rowBadge];
        if (aliases[badge]) {
            badge = aliases[badge];
        }
        const [selected, setSelected] = useState(false);
        return (
            <Accordion.Item eventKey={idx} className={selected && 'selected-result'} key={idx}>
                <Accordion.Header className="my-accordion-header" onClick={() => setSelected(!selected)}>
                    <div>{title}</div>
                    <div className="squashed-row">
                        <div className="accordion-badge">{badge}</div>
                        <div className="expand-symbol">&#x25BC;</div>
                    </div>
                </Accordion.Header>
                <Accordion.Body className="my-accordion-body">
                    <div className="panel-sub">result #{idx + 1}</div>
                    <AgnosticTable keyValues={Object.entries(result)} toIgnore={new Set(['address', 'id'])} aliases={aliases} />
                </Accordion.Body>
            </Accordion.Item>
        );
    }

    return ( 
        <div >
            {sectionHeader && sectionHeader} 
            {SortAndFilters && SortAndFilters}
            {!noResults && Pagination && Pagination}

            <div >
                {(loading || noResults) && <div className="results">

                    {loading ? <div id="spinner" className="justify-content-center">
                        <div className="spinner-border" role="status">
                            <span className="sr-only">Loading...</span>
                        </div>
                    </div> : (noResults && noDataComponent)
                    }
                </div>}
                <Accordion className="overall-accordion" >
                    {results.map((result, idx) => <Row key={idx} result={result} idx={idx}></Row>)}
                </Accordion>
            </div>
        </div>
    )
}
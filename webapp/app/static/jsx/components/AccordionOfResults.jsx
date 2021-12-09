import React, { useState } from 'react';
import { Accordion, Toast, } from 'react-bootstrap';
import AgnosticTable from './AgnosticTable';
import { CopyToClipboard } from 'react-copy-to-clipboard';

const Spinner = (<div id="spinner" className="justify-content-center">
    <div className="spinner-border" role="status">
        <span className="sr-only">Loading...</span>
    </div>
</div>)

export default function AccordionOfResults(props) {
    const { results, loading, aliases, rowTitle, rowBadge, sectionHeader,
        noDataComponent, SortAndFilters, Pagination, startIndex=0 } = props;
    const noResults = results.length == 0;
    
    function Row({ result, idx }) {
        const title = result[rowTitle];
        let badge = result[rowBadge];
        if (aliases[badge]) {
            badge = aliases[badge];
        }

        const [showToast, setShowToast] = useState(false);

        const [selected, setSelected] = useState(false);
        const expandable = Object.keys(result).length > 2;
        return (
            <Accordion.Item eventKey={idx} className={`col-12 ${selected && 'selected-result'}`} key={idx}>
                <div className="row">
                    <Accordion.Header className="col-12 my-accordion-header" onClick={() => setSelected(!selected)}>
                        <div className="row">
                            <div className="col-12 first-part-accordion-header">
                                {title}
                                &nbsp;
                                <CopyToClipboard text={title} onCopy={() => setShowToast(true)}><i className="far fa-copy"></i></CopyToClipboard>
                                &nbsp;&nbsp;<Toast className="copied-badge" width="100" onClose={() => setShowToast(false)} show={showToast} delay={3000} autohide>
                                    Copied!
                                </Toast>
                            </div>

                            <div className="col-12 squashed-row">
                                <div className="accordion-badge">{badge}</div>
                                {expandable && <div className="expand-symbol">&#x25BC;</div>}
                            </div>
                        </div>
                    </Accordion.Header>
                    {expandable &&
                        <Accordion.Body className="my-accordion-body">
                            <div className="panel-sub">result #{startIndex + idx + 1}</div>
                            <AgnosticTable keyValues={result} toIgnore={new Set(['address', 'id'])} aliases={aliases} />
                            <div className="etherscan-link">
                                {rowTitle === 'address' && <a href={`https://etherscan.io/address/${title}`}>view on etherscan</a>}
                            </div>
                        </Accordion.Body>}
                </div>
            </Accordion.Item>
        );
    }

    return (
        <div className="row linked-adress">
            {sectionHeader && sectionHeader}
            {SortAndFilters && SortAndFilters}
            {!noResults && Pagination && Pagination}

            <div className="col-12">
                {(loading || noResults) && <div className="results loading col-12">

                    {loading ? Spinner : (noResults && noDataComponent)
                    }
                </div>}
                <Accordion className="row overall-accordion" >
                    {results.map((result, idx) => <Row key={idx} result={result} idx={idx}></Row>)}

                </Accordion>

            </div>
            <div className="results loading col-12">
                {loading && Spinner }
            </div>
            {!noResults && Pagination && Pagination}
        </div>
    )
}
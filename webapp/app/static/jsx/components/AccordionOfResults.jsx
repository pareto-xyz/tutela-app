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
        noDataComponent, SortAndFilters, Pagination, startIndex = 0,
        myClassName = "default-accordion-class", } = props;
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
            <Accordion.Item eventKey={idx} className={`row ${selected && 'selected-result'}`} key={idx}>
                <Accordion.Header className="col-12 my-accordion-header" onClick={() => setSelected(!selected)}>
                    <div className="row address-row">
                        <div className="col-12">
                            <div className="row row-address">
                                <div className="col-7 col-sm-6 col-md-5 col-lg-3">
                                    <div className="mt-1r accordion-badge">{badge}</div>
                                </div>
                                <div className="col-12 col-lg-8">
                                    <div className="row address-copy">
                                        <div className="text col-9">{title}</div>
                                        <div className="copy col-2">
                                            <CopyToClipboard text={title} onCopy={() => setShowToast(true)}><i className="far fa-copy"></i></CopyToClipboard>
                                            <Toast className="copied-badge" onClose={() => setShowToast(false)} show={showToast} delay={3000} autohide>
                                                Copied!
                                            </Toast>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {expandable && <div className="col-1 mt-1r expand-symbol"><i className="fas fa-angle-down"></i></div>}
                    </div>
                </Accordion.Header>
                <div className="col-12 drop-info">
                    <div className="row">
                        {expandable &&
                            <Accordion.Body className="col-12">
                                <div className="row">
                                    <div className="col-12">
                                        <div className="my-accordion-body">
                                            <div className="row">
                                                <div className="col-12 panel-sub">result #{startIndex + idx + 1}</div>
                                                <div className="col-12 table-responsive">
                                                    <AgnosticTable keyValues={result} toIgnore={new Set(['conf', 'address', 'id', 'metadata'])} aliases={aliases} />
                                                </div>
                                                <div className="col-12 etherscan-link">
                                                    {rowTitle === 'address' && <a href={`https://etherscan.io/address/${title}`}>view on etherscan</a>}
                                                    {rowTitle === 'transaction' && <a href={`https://etherscan.io/tx/${title}`}>view on etherscan</a>}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </Accordion.Body>}
                    </div>
                </div>
            </Accordion.Item>
        );
    }

    return (
        <div className="row">
            <div className="col-12">
                <div className={myClassName}>
                    <div className="row">
                        {sectionHeader && sectionHeader}
                        {SortAndFilters && SortAndFilters}
                        {!noResults && Pagination && Pagination}
                        {(loading || noResults) && <div className="results loading col-12">

                            {loading ? Spinner : (noResults && noDataComponent)
                            }
                        </div>}
                        <Accordion className="col-12 overall-accordion" >
                            {results.map((result, idx) => <Row key={idx} result={result} idx={idx}></Row>)}
                        </Accordion>
                        <div className="results loading col-12">
                            {loading && Spinner}
                        </div>
                        {!noResults && Pagination && Pagination}
                    </div>
                </div>
            </div>
        </div>
    )
}
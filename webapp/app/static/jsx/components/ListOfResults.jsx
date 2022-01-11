import React, { useState } from 'react';
import { ListGroup, Toast, } from 'react-bootstrap';
import { CopyToClipboard } from 'react-copy-to-clipboard';

const Spinner = (<div id="spinner" className="justify-content-center">
    <div className="spinner-border" role="status">
        <span className="sr-only">Loading...</span>
    </div>
</div>)

export default function ListOfResults(props) {
    const { results, loading, aliases, rowTitle, rowBadge, sectionHeader,
        noDataComponent, } = props;
    const noResults = results.length == 0;
    
    function Row({ result, idx }) {
        const title = result[rowTitle];
        let badge = result[rowBadge];
        if (aliases[badge]) {
            badge = aliases[badge];
        }

        const [showToast, setShowToast] = useState(false);

        const [selected, setSelected] = useState(false);
        return (
            <div className={`row ${selected && 'selected-result'}`} key={idx}>
                <ListGroup.Item className="col-12 my-accordion-header" onClick={() => setSelected(!selected)}>
                    <div className="row adress-row">
                        <div className="col-10">
                            <div className="row">
                                <div className="col-md-5 col-lg-3 mt-1r accordion-badge">{badge}</div>
                                <div className="col-md-12 col-lg-9 mt-1r first-part-accordion-header">
                                    <div className="adress-copy row">
                                        <div className="text col-10">{title}</div>
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
                    </div>
                </ListGroup.Item>
               
            </div>
        );
    }

    return (
        <div className="row">
            <div className="col-12">
                <div >
                    <div className="row">
                        {sectionHeader && sectionHeader}
                        {(loading || noResults) && <div className="results loading col-12">

                            {loading ? Spinner : (noResults && noDataComponent)
                            }
                        </div>}
                        <ListGroup className="col-12 overall-accordion" >
                            {results.map((result, idx) => <Row key={idx} result={result} idx={idx}></Row>)}
                        </ListGroup>
                        <div className="results loading col-12">
                            {loading && Spinner }
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
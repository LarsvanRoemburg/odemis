

# By default, not all XmlResourceHandlers are present in the XML resource
# generated by the XRCED program (the main_xrc.py module, in Odemis' case).
# To handle custom and the more exotic controls, the approriate handlers should
# be added to the resource. We do this by replacing the default `get_resources`
# function in the main_xrc.py module with the function in this module. The
# replacement should take place before any references are made to the frames,
# dialog and controls defined within the main_xrc.py module.

import odemis.gui.main_xrc

def odemis_get_resources():
    """ This function provides access to the XML handlers needed for
        non-standard controls defined in the XRC file.
    """
    if odemis.gui.main_xrc.__res == None:    #pylint: disable=W0212
        from odemis.gui.xmlh.xh_delmic import FoldPanelBarXmlHandler
        odemis.gui.main_xrc.__init_resources() #pylint: disable=W0212
        odemis.gui.main_xrc.__res.InsertHandler(FoldPanelBarXmlHandler()) #pylint: disable=W0212
    return odemis.gui.main_xrc.__res #pylint: disable=W0212

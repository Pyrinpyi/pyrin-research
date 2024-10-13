import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:intl/intl.dart";
import "package:provider/provider.dart";
import "package:pyrin_app/components/tokens_list.dart";
import "package:pyrin_app/core/section.dart";
import "package:pyrin_app/core/wallet_provider.dart";
import "package:pyrin_app/modals/confirmation.dart";
import "package:pyrin_app/ui.dart";

class SwapTextField extends StatefulWidget
{
  final String? value;
  final Function(String)? onChanged;

  SwapTextField({this.value, this.onChanged});

  @override
  State<SwapTextField> createState() => _SwapTextFieldState();
}

class _SwapTextFieldState extends State<SwapTextField> {
  TextEditingController controller = TextEditingController();


  @override
  void initState()
  {
    super.initState();
    controller.addListener(()
    {
      if (widget.onChanged != null)
      {
        widget.onChanged!(controller.text);
      }
    });
  }

  @override
  Widget build(BuildContext context)
  {
    return Container(
        decoration: BoxDecoration(
          color: PyrinColors.BLACK1_COLOR,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: PyrinTextField(
                controller: controller,
                hintText: "0.0",
                keyboardType: TextInputType.number,
              ),
            ),
            Container(
              margin: const EdgeInsets.only(right: 10),
              child: PyrinDropdown(
                value: widget.value,
                items:
                [
                  "PYI",
                  "PYX",
                  "PYC",
                ],
                builder: (context, value, child)
                {
                  return Row(
                    children: [
                      TokenIcon(symbol: value),
                      child ?? Container(),
                    ],
                  );
                },
              ),
            ),
          ],
        )
    );
  }
}


class SwapSection extends StatefulWidget
{
  @override
  State<SwapSection> createState() => _SwapSectionState();
}

class _SwapSectionState extends State<SwapSection>
{
  String from = "PYI";
  String to = "PYX";

  double fromAmount = 0.0;
  double toAmount = 0.0;

  @override
  Widget build(BuildContext context)
  {
    final int fromValue = from.split(""). map((v) => v.codeUnitAt(0)).reduce((a, b) => a + b);
    final int toValue = to.split(""). map((v) => v.codeUnitAt(0)).reduce((a, b) => a + b);

    return SectionContainer(
        name: "Swap",
        child: ListView(
          padding: const EdgeInsets.all(20),
          shrinkWrap: true,
          children: [
            const SizedBox(height: 40),
            PyrinGroup(
                label: "You Pay",
                footer: "Balance: ${NumberFormat("#,##0.00", "en_US").format(fromValue * 50)} $from",
                child: SwapTextField(
                  value: from,
                  onChanged: (value) => setState(() => fromAmount = double.parse(value)), // TODO: When happen on error
                )
            ),
            Container(
              margin: const EdgeInsets.symmetric(vertical: 20),
              child: CircleIconButton(icon: "switch", onClick: onSwitchClick),
            ),
            PyrinGroup(
                label: "You Receive",
                footer: "Balance: ${NumberFormat("#,##0.00", "en_US").format(toValue * 50)} $to",
                child: SwapTextField(
                  value: to,
                  onChanged: (value) => setState(() => toAmount = double.parse(value)), // TODO: When happen on error
                )
            ),
            Container(
                margin: const EdgeInsets.symmetric(vertical: 20),
                child: Text(
                  "${1} $from ~= ${NumberFormat("#,##0.00000", "en_US").format(fromValue / toValue)} $to",
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium!.copyWith(
                      fontSize: 16,
                      fontWeight: FontWeight.w400,
                      color: PyrinColors.TEXT_COLOR
                  ),
                )
            ),
            const SizedBox(height: 40),
            PyrinElevatedButton(
              text: "Swap",
              onClick: onSwapClick,
              wide: true,
            ),
          ],
        )
    );
  }

  onSwitchClick()
  {
    setState(()
    {
      String s = from;
      from = to;
      to = s;
    });
  }

  onSwapClick()
  {
    final String receiveAddress = Provider.of<WalletProvider>(context, listen: false).receiveAddress;

    showConfirmationModal(
      context: context,
      text: "Swap Transaction",
      fromAddress: receiveAddress,
      toAddress: receiveAddress,
      items: [
        ConfirmationItem(
          name: "Network fees",
          child: Text("0.001 $from"),
        ),
        ConfirmationItem(
          name: "Total",
          child: Text("${fromAmount + 0.001} $from"),
        ),
      ],
      onConfirm: ()
      {
        Navigator.pop(context);

        transactionConfirmedModal(context);
      },
      tokens: [from, to],
    );
  }
}

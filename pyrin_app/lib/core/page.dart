import "package:flutter/material.dart";
import "package:flutter_svg/flutter_svg.dart";
import "package:pyrin_app/ui.dart";

class RoutePage extends StatefulWidget
{
    final String name;
    final Widget child;
    final List<Widget>? buttons;

    const RoutePage({super.key, required this.name, required this.child, this.buttons});

    @override
    State<RoutePage> createState() => RoutePageState();
}


class RoutePageState extends State<RoutePage>
{
  @override
  Widget build(BuildContext context)
  {
      return Scaffold(
        body: SafeArea(
          child: Container(
            padding: EdgeInsets.all(20).copyWith(top: 0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  margin: EdgeInsets.only(bottom: 20),
                  height: 40,
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      Align(
                        alignment: Alignment.centerLeft,
                        child: IconButton(
                        icon: SvgPicture.asset("assets/icons/arrow-left.svg"),
                          onPressed: () => Navigator.pop(context),
                        ),
                      ),
                      Center(child: Text(widget.name, style: Theme.of(context).textTheme.bodyLarge!.copyWith(fontWeight: FontWeight.w600))),
                    ],
                  ),
                ),
                Expanded(
                  child: Container(
                      width: double.infinity,
                      child: widget.child
                  ),
                ),
                ...widget.buttons != null ? [
                  const SizedBox(height: 20),
                  Container(
                    width: double.infinity,
                    child: Row(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.center,
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: widget.buttons!.map((button) => Flexible(child: button)).toList()
                    ),
                  ),
                ] : [],
              ],
            ),
          ),
        ),
      );
  }
}
